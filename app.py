import io
import re
import csv
import json
import sqlite3
import tempfile
import datetime
import os
from collections import defaultdict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from transformers import pipeline
from docx import Document
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ============================================================================
# SECTION 1: NLP PIPELINE (NER, summarization, action/decision extraction)
# ============================================================================

ACTION_VERBS = {
    "fix", "prepare", "complete", "finish", "build", "test", "review",
    "send", "update", "create", "deploy", "write", "schedule",
    "follow up", "investigate", "implement", "check", "share", "finalize",
}

ACTION_CUES = [
    r"\bwill\b", r"\bneeds? to\b", r"\bshould\b", r"\bhas to\b",
    r"\bis going to\b", r"\bassigned to\b", r"\bresponsible for\b",
    r"\bto do\b", r"\baction item\b", r"\bplease\b",
]

DECISION_CUES = [
    r"\bdecided\b", r"\bwe agreed\b", r"\bagreed to\b", r"\bapproved\b",
    r"\bfinal(ised|ized) decision\b", r"\bconcluded\b", r"\bwill go with\b",
]

TECH_KEYWORDS = {
    "api", "database", "frontend", "backend", "bug", "login", "server",
    "deployment", "testing", "release", "sprint", "homepage", "ui", "ux",
    "app", "website", "pipeline", "model", "streamlit", "docker", "cloud",
}


@st.cache_resource(show_spinner=False)
def load_spacy():
    """
    Load the spaCy NER model. It must already be installed as a pip
    package via requirements.txt (see the wheel URL there) — Streamlit
    Cloud's environment is read-only at runtime, so attempting to
    spacy.cli.download() it here as a fallback would fail with a
    'Permission denied' error and silently crash the app.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error(
            "The spaCy language model 'en_core_web_sm' isn't installed. "
            "Add this line to requirements.txt and redeploy:\n\n"
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """
    Lightweight distilled BART summarizer (~300MB, deploy-friendly).
    Loaded directly via AutoTokenizer/AutoModelForSeq2SeqLM + .generate()
    instead of pipeline("summarization", ...) so it works whether
    transformers v4 or v5 ends up installed.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _run_summarizer(text: str, max_len: int, min_len: int) -> str:
    tokenizer, model = load_summarizer()
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def clean_transcript(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str, nlp=None):
    nlp = nlp or load_spacy()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def extract_entities(text: str) -> dict:
    nlp = load_spacy()
    doc = nlp(text)

    people, orgs, dates = set(), set(), set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.add(ent.text.strip())
        elif ent.label_ == "ORG":
            orgs.add(ent.text.strip())
        elif ent.label_ in ("DATE", "TIME"):
            dates.add(ent.text.strip())

    lower = text.lower()
    tech = {kw for kw in TECH_KEYWORDS if kw in lower}

    return {
        "people": sorted(people),
        "organizations": sorted(orgs),
        "dates": sorted(dates),
        "projects_tech": sorted(tech),
    }


def summarize_text(text: str, max_len: int = 130, min_len: int = 30) -> str:
    if len(text.split()) < 40:
        return text.strip()

    words = text.split()
    chunk_size = 700
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    partial_summaries = []
    for chunk in chunks:
        try:
            partial_summaries.append(_run_summarizer(chunk, max_len, min_len))
        except Exception:
            partial_summaries.append(chunk[:300])

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = " ".join(partial_summaries)
    return _run_summarizer(combined, max_len, min_len)


def extract_key_points(text: str, top_n: int = 6) -> list:
    nlp = load_spacy()
    doc = nlp(text)
    sentences = [s for s in doc.sents if len(s.text.split()) > 4]

    scored = []
    for sent in sentences:
        score = len(sent.ents)
        score += sum(1 for tok in sent if tok.text.lower() in TECH_KEYWORDS)
        scored.append((score, sent.text.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen, points = set(), []
    for _, text_s in scored:
        key = text_s[:40]
        if key not in seen:
            points.append(text_s)
            seen.add(key)
        if len(points) >= top_n:
            break
    return points


def extract_decisions(text: str) -> list:
    sentences = split_sentences(text)
    decisions = []
    for s in sentences:
        if any(re.search(cue, s, re.IGNORECASE) for cue in DECISION_CUES):
            decisions.append(s)
    return decisions


FIRST_PERSON_RE = re.compile(
    r"\b(i will|i'll|i am going to|i'm going to|i should|i need to|i have to|i can)\b", re.IGNORECASE
)


def extract_action_items(segments: list, known_speakers: list = None) -> list:
    """
    Works one speaker turn at a time, so every task knows who said it.
    Owner rules: "Name, please ..." -> Name; "I will ..." -> the speaker;
    otherwise a name mentioned in the sentence; otherwise the speaker.
    Questions, decisions and past-tense statements are not tasks.
    """
    nlp = load_spacy()
    known = [s for s in (known_speakers or []) if s != "Unknown Speaker"]
    items = []

    for seg in segments:
        speaker = seg["speaker"] if seg["speaker"] != "Unknown Speaker" else None
        doc = nlp(seg["text"])
        sents = list(doc.sents)

        for idx, sent in enumerate(sents):
            s_text = sent.text.strip()
            if not s_text or s_text.endswith("?"):
                continue
            if any(re.search(c, s_text, re.IGNORECASE) for c in DECISION_CUES):
                continue

            has_cue = any(re.search(cue, s_text, re.IGNORECASE) for cue in ACTION_CUES)
            has_verb = any(re.search(rf"\b{re.escape(v)}\b", s_text, re.IGNORECASE) for v in ACTION_VERBS)
            if not (has_cue or has_verb):
                continue

            addressee = next(
                (n for n in known if n != speaker and re.match(rf"\s*{re.escape(n)}\s*,", s_text, re.IGNORECASE)),
                None,
            )
            if addressee:
                person = addressee
            elif FIRST_PERSON_RE.search(s_text) and speaker:
                person = speaker
            else:
                ent_people = [e.text for e in sent.ents if e.label_ == "PERSON"]
                named = next((n for n in known if re.search(rf"\b{re.escape(n)}\b", s_text)), None)
                person = named or (ent_people[0] if ent_people else None) or speaker or "Unassigned"

            dates = [e.text for e in sent.ents if e.label_ in ("DATE", "TIME")]
            deadline = dates[0] if dates else "Not specified"

            # priority: this sentence first, else a neighbouring sentence in the same turn
            priority = detect_priority(s_text)
            if priority == "Medium":
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(sents):
                        p = detect_priority(sents[j].text)
                        if p != "Medium":
                            priority = p
                            break

            items.append({
                "person": person,
                "task": s_text,
                "deadline": deadline,
                "priority": priority,
            })

    return items


# ============================================================================
# SECTION 2: PRIORITY DETECTION (keyword-based, fast & deploy-friendly)
# ============================================================================

URGENT_WORDS = {"urgent", "asap", "immediately", "critical", "blocker", "today", "tomorrow"}
LOW_WORDS = {"eventually", "whenever", "someday", "nice to have", "low priority",
             "not urgent", "not blocking", "not critical", "no rush", "no hurry"}


def detect_priority(task_text: str) -> str:
    # Keyword-based on purpose (keeps memory low on Streamlit's free tier).
    # Low/negated phrases ("not urgent") are checked FIRST, otherwise the
    # word "urgent" inside "not urgent" would wrongly give High.
    t = task_text.lower()
    if any(w in t for w in LOW_WORDS):
        return "Low"
    if any(re.search(rf"\b{w}\b", t) for w in URGENT_WORDS):
        return "High"
    return "Medium"


def annotate_priorities(action_items: list) -> list:
    # keep the priority already decided during extraction; only fill gaps
    for item in action_items:
        item["priority"] = item.get("priority") or detect_priority(item.get("task", ""))
    return action_items


# ============================================================================
# SECTION 3: SPEAKER IDENTIFICATION (label parsing)
# ============================================================================

SPEAKER_PATTERN = re.compile(
    r"^\s*[\[\(]?([A-Z][a-zA-Z .]{1,30})[\]\)]?\s*[:\-]\s*(.+)$"
)


def parse_speakers(transcript: str):
    segments = []
    speaker_counts = {}
    char_cursor = 0

    for line in transcript.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            char_cursor += len(line) + 1
            continue

        match = SPEAKER_PATTERN.match(line_stripped)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
        else:
            speaker = "Unknown Speaker"
            text = line_stripped

        segments.append({
            "speaker": speaker,
            "text": text,
            "char_start": char_cursor,
        })
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        char_cursor += len(line) + 1

    return segments, speaker_counts


def build_speaker_char_map(segments):
    return {seg["char_start"]: seg["speaker"] for seg in segments}


def plain_transcript(segments):
    return " ".join(seg["text"] for seg in segments)


# ============================================================================
# SECTION 4: Q&A + SEARCH
# ============================================================================

MY_TASK_PATTERNS = [
    r"my task", r"assigned to me", r"what do i (have to|need to) do",
    r"what.?s my", r"tasks for me",
]

DECISION_QUESTION_PATTERNS = [
    r"\bdecide[ds]?\b", r"\bdecision[s]?\b", r"\bagreed?\b", r"\bconclu(ded|sion)\b",
]

SUMMARY_QUESTION_PATTERNS = [
    r"\bsummar(y|ize|ise)\b", r"\bwhat happened\b", r"\bwhat was the meeting about\b",
    r"\boverview\b", r"\bwhat was discussed\b",
]

QUESTION_STOPWORDS = {
    "what", "did", "the", "was", "were", "about", "that", "this", "have", "has",
    "does", "do", "who", "when", "where", "why", "how", "for", "and", "are",
    "decide", "decided", "decision", "agreed", "you", "we", "they", "with",
}


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def is_my_tasks_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in MY_TASK_PATTERNS)


def answer_my_tasks(action_items: list, user_name: str) -> str:
    if not user_name:
        return "Tell me your name first (see the sidebar) so I can match your tasks."

    mine = [
        item for item in action_items
        if item.get("person", "").strip().lower() == user_name.strip().lower()
    ]
    if not mine:
        return f"I couldn't find any action items assigned to **{user_name}** in this meeting."

    lines = [f"Here's what's assigned to **{user_name}**:\n"]
    for item in mine:
        lines.append(f"- {item['task']} (Deadline: {item['deadline']}, Priority: {item.get('priority', 'N/A')})")
    return "\n".join(lines)


def _keyword_overlap(question: str, sentence: str) -> int:
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower())) - QUESTION_STOPWORDS
    s_words = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
    return len(q_words & s_words)


def _best_matches(question: str, items: list, min_overlap: int = 1, top_n: int = 3) -> list:
    scored = [(_keyword_overlap(question, s), s) for s in items]
    scored = [x for x in scored if x[0] >= min_overlap]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_n]]


def answer_question(
    question: str, transcript: str, action_items: list, user_name: str = "",
    decisions: list = None, summary: str = None,
) -> str:
    if is_my_tasks_question(question):
        return answer_my_tasks(action_items, user_name)

    q_lower = question.lower()
    decisions = decisions or []

    if any(re.search(p, q_lower) for p in DECISION_QUESTION_PATTERNS):
        matches = _best_matches(question, decisions)
        if matches:
            return "Here's what was decided:\n\n" + "\n".join(f"- {m}" for m in matches)
        if decisions:
            return "Here's everything that was decided in this meeting:\n\n" + "\n".join(f"- {d}" for d in decisions)
        return "I didn't detect any explicit decisions in this meeting."

    if any(re.search(p, q_lower) for p in SUMMARY_QUESTION_PATTERNS) and summary:
        return summary

    try:
        qa = load_qa_model()
        result = qa(question=question, context=transcript)
        answer = result.get("answer", "").strip()
        score = result.get("score", 0)
        if not answer or score < 0.05:
            return "I couldn't find a confident answer to that in the transcript. Try rephrasing, or search the transcript directly."
        return answer
    except Exception:
        return "The Q&A model couldn't process that right now. Try again or use the transcript search instead."


def search_transcript(transcript: str, query: str, context_chars: int = 60) -> list:
    if not query.strip():
        return []

    results = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for match in pattern.finditer(transcript):
        start = max(0, match.start() - context_chars)
        end = min(len(transcript), match.end() + context_chars)
        snippet = transcript[start:end].replace("\n", " ")
        highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)
        results.append(f"...{highlighted}...")

    return results


# ============================================================================
# SECTION 6: DATABASE (meeting history, SQLite)
# ============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meetings.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT,
            transcript TEXT,
            summary TEXT,
            data_json TEXT
        )
    """)
    conn.commit()
    return conn


def save_meeting(title: str, transcript: str, summary: str, data: dict) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO meetings (title, created_at, transcript, summary, data_json) VALUES (?, ?, ?, ?, ?)",
        (title, datetime.datetime.now().isoformat(timespec="seconds"), transcript, summary, json.dumps(data)),
    )
    conn.commit()
    meeting_id = cur.lastrowid
    conn.close()
    return meeting_id


def list_meetings() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at, summary FROM meetings ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2], "summary": r[3]}
        for r in rows
    ]


def get_meeting(meeting_id: int) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, created_at, transcript, summary, data_json FROM meetings WHERE id = ?",
        (meeting_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "created_at": row[2],
        "transcript": row[3],
        "summary": row[4],
        "data": json.loads(row[5]),
    }


def delete_meeting(meeting_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.commit()
    conn.close()


# ============================================================================
# SECTION 7: EXPORT UTILITIES (CSV, DOCX, PDF)
#   NOTE: calendar (.ics) export was removed on request.
# ============================================================================

def action_items_to_csv_bytes(action_items: list) -> bytes:
    df = pd.DataFrame(action_items)
    cols = [c for c in ["person", "task", "deadline", "priority", "status"] if c in df.columns]
    df = df[cols] if cols else df
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue().encode("utf-8")


def report_to_docx_bytes(meeting_title, summary, key_points, decisions, action_items, entities) -> bytes:
    doc = Document()
    doc.add_heading(f"Meeting Report: {meeting_title}", level=1)

    doc.add_heading("Meeting Summary", level=2)
    doc.add_paragraph(summary or "N/A")

    doc.add_heading("Key Discussion Points", level=2)
    for p in key_points:
        doc.add_paragraph(p, style="List Bullet")

    doc.add_heading("Decisions Made", level=2)
    for d in decisions:
        doc.add_paragraph(d, style="List Bullet")

    doc.add_heading("Action Items", level=2)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "Person", "Task", "Deadline", "Priority"
    for item in action_items:
        row = table.add_row().cells
        row[0].text = str(item.get("person", ""))
        row[1].text = str(item.get("task", ""))
        row[2].text = str(item.get("deadline", ""))
        row[3].text = str(item.get("priority", ""))

    doc.add_heading("Important Entities", level=2)
    doc.add_paragraph(f"People: {', '.join(entities.get('people', [])) or 'N/A'}")
    doc.add_paragraph(f"Organizations: {', '.join(entities.get('organizations', [])) or 'N/A'}")
    doc.add_paragraph(f"Dates: {', '.join(entities.get('dates', [])) or 'N/A'}")
    doc.add_paragraph(f"Projects/Tech: {', '.join(entities.get('projects_tech', [])) or 'N/A'}")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def report_to_pdf_bytes(meeting_title, summary, key_points, decisions, action_items, entities) -> bytes:
    """
    FIX: fpdf2's multi_cell(), when called with width=0 (i.e. "use the full
    remaining line width"), leaves the cursor sitting at the RIGHT margin
    afterwards instead of resetting it to the left. The next multi_cell(0, ...)
    call then computes its available width as (cursor_x -> right_margin),
    which is ~0, and fpdf2 raises "Not enough horizontal space to render a
    single character". Passing new_x=XPos.LMARGIN, new_y=YPos.NEXT to every
    multi_cell() call explicitly resets the cursor to the next line's left
    margin, which is what actually fixes it (this was tested end-to-end).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def clean(t):
        t = str(t).encode("latin-1", "replace").decode("latin-1")
        # fpdf2's line-wrapper crashes (FPDFException) if it hits a single
        # "word" (no whitespace) too wide for the page width — e.g. a
        # run-on token from an imperfect model-generated summary. Force
        # soft breaks into any very long unbroken run so wrapping always
        # succeeds instead of raising.
        return re.sub(r"\S{40,}", lambda m: " ".join(m.group(0)[i:i + 40] for i in range(0, len(m.group(0)), 40)), t)

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, clean(f"Meeting Report: {meeting_title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    def section(title, body_lines):
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        for line in body_lines:
            pdf.multi_cell(0, 6, clean(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Meeting Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, clean(summary or "N/A"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    section("Key Discussion Points", key_points or ["N/A"])
    section("Decisions Made", decisions or ["N/A"])

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Action Items", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for item in action_items:
        line = f"{item.get('person','')} -> {item.get('task','')} (Due: {item.get('deadline','')}, Priority: {item.get('priority','')})"
        pdf.multi_cell(0, 6, clean(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Important Entities", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, clean(f"People: {', '.join(entities.get('people', [])) or 'N/A'}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 6, clean(f"Organizations: {', '.join(entities.get('organizations', [])) or 'N/A'}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 6, clean(f"Dates: {', '.join(entities.get('dates', [])) or 'N/A'}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 6, clean(f"Projects/Tech: {', '.join(entities.get('projects_tech', [])) or 'N/A'}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


# ============================================================================
# SECTION 8: EMAIL DRAFT GENERATION
# ============================================================================

def group_items_by_person(action_items: list) -> dict:
    grouped = defaultdict(list)
    for item in action_items:
        grouped[item.get("person", "Unassigned")].append(item)
    return grouped


def generate_email_draft(person: str, items: list, meeting_title: str) -> dict:
    subject = f"Action Items for you from: {meeting_title}"
    lines = [f"Hi {person},", "", f"Here are your action items from \"{meeting_title}\":", ""]
    for item in items:
        lines.append(f"- {item['task']}")
        lines.append(f"  Deadline: {item.get('deadline', 'Not specified')} | Priority: {item.get('priority', 'N/A')}")
    lines += ["", "Please confirm once completed.", "", "Thanks,", "Meeting Bot"]
    return {"to": person, "subject": subject, "body": "\n".join(lines)}


def generate_all_drafts(action_items: list, meeting_title: str) -> list:
    grouped = group_items_by_person(action_items)
    return [
        generate_email_draft(person, items, meeting_title)
        for person, items in grouped.items()
        if person and person != "Unassigned"
    ]


# ============================================================================
# SECTION 9: OPTIONAL AUDIO TRANSCRIPTION (free, online STT)
# ============================================================================

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False


def audio_file_to_wav(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    wav_path = tmp_in_path + ".wav"
    audio = AudioSegment.from_file(tmp_in_path)
    audio.export(wav_path, format="wav")
    os.remove(tmp_in_path)
    return wav_path


def transcribe_wav(wav_path: str, chunk_seconds: int = 55) -> str:
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    chunk_ms = chunk_seconds * 1000

    transcript_parts = []
    for start in range(0, duration_ms, chunk_ms):
        chunk = audio[start:start + chunk_ms]
        chunk_path = wav_path + f".chunk_{start}.wav"
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            transcript_parts.append(text)
        except sr.UnknownValueError:
            transcript_parts.append("[inaudible]")
        except sr.RequestError as e:
            transcript_parts.append(f"[speech recognition service error: {e}]")
        finally:
            os.remove(chunk_path)

    os.remove(wav_path)
    return " ".join(transcript_parts)


# ============================================================================
# SECTION 10: STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Meeting-to-Action-Items AI", page_icon="🎤", layout="wide")

# ---------------------------------------------------------------- session state
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# THE FIX: both input modes now write into ONE session_state-backed key
# ("current_transcript") instead of a plain local variable. A local
# variable resets to "" on every script rerun (Streamlit reruns the
# whole script on every click), which is what was wiping the
# transcribed audio text the moment "Analyze Meeting" was clicked.
# session_state persists across reruns, so this survives.
if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎤 Meeting AI")
    st.caption("NLP-powered meeting summarizer & task extractor")
    page = st.radio(
        "Navigate",
        ["🆕 New Meeting", "📊 Dashboard", "📜 History", "📤 Export"],
    )
    st.divider()
    st.caption("Built with spaCy NER + DistilBART summarization + "
               "keyword-based priority detection.")


ANALYSIS_VERSION = 2


def refresh_if_stale(data: dict) -> dict:
    """
    Analyses saved (or kept in memory) by an older version of the app have every
    task marked 'Unassigned', which is why no email drafts appear. Rebuild the
    action items from the stored transcript so old data works without re-analyzing.
    """
    if data.get("version") == ANALYSIS_VERSION:
        return data
    transcript = clean_transcript(data.get("transcript", ""))
    if not transcript:
        return data
    segments, speaker_counts = parse_speakers(transcript)
    items = annotate_priorities(extract_action_items(segments, known_speakers=list(speaker_counts)))
    for item in items:
        item["status"] = "Pending"
    data["action_items"] = items
    data["speaker_counts"] = speaker_counts
    entities = data.setdefault("entities", {})
    entities["people"] = sorted(
        set(entities.get("people", [])) | {s for s in speaker_counts if s != "Unknown Speaker"}
    )
    data["version"] = ANALYSIS_VERSION
    return data


def run_pipeline(transcript_raw: str, meeting_title: str):
    transcript_raw = clean_transcript(transcript_raw)
    segments, speaker_counts = parse_speakers(transcript_raw)
    char_map = build_speaker_char_map(segments)
    plain_text = plain_transcript(segments) if segments else transcript_raw

    with st.spinner("Summarizing meeting..."):
        summary = summarize_text(plain_text)
    with st.spinner("Extracting entities..."):
        entities = extract_entities(plain_text)
    entities["people"] = sorted(
        set(entities["people"]) | {s for s in speaker_counts if s != "Unknown Speaker"}
    )
    with st.spinner("Identifying key discussion points..."):
        key_points = extract_key_points(plain_text)
    with st.spinner("Detecting decisions..."):
        decisions = extract_decisions(plain_text)
    with st.spinner("Extracting action items..."):
        action_items = extract_action_items(segments, known_speakers=list(speaker_counts))
    with st.spinner("Scoring priority..."):
        action_items = annotate_priorities(action_items)
    for item in action_items:
        item["status"] = "Pending"

    data = {
        "title": meeting_title,
        "transcript": transcript_raw,
        "plain_text": plain_text,
        "summary": summary,
        "entities": entities,
        "key_points": key_points,
        "decisions": decisions,
        "action_items": action_items,
        "speaker_counts": speaker_counts,
        "version": ANALYSIS_VERSION,
    }
    st.session_state.analysis = data
    save_meeting(meeting_title, transcript_raw, summary, data)
    return data


# ---------------------------------------------------------------- New Meeting
if page == "🆕 New Meeting":
    st.header("New Meeting")
    meeting_title = st.text_input("Meeting title", value=f"Meeting {datetime.date.today()}")

    input_mode = st.radio("Input type", ["📄 Paste / upload transcript", "🎧 Upload audio"], horizontal=True)

    if input_mode == "📄 Paste / upload transcript":
        uploaded = st.file_uploader("Upload a .txt transcript (optional)", type=["txt"])
        if uploaded:
            # a freshly uploaded file overwrites whatever was there before
            st.session_state.current_transcript = uploaded.read().decode("utf-8", errors="ignore")

        # key=... binds this box directly to session_state["current_transcript"];
        # no `value=` needed (and none should be passed alongside a key,
        # to avoid the "default value AND Session State" warning).
        st.text_area(
            "Or paste transcript here (tip: prefix lines with 'Name: ' for speaker identification)",
            height=280,
            key="current_transcript",
            placeholder="Dharshini: I'll fix the login bug by Friday.\nPriya: I'll prepare test cases by Thursday.\n...",
        )
    else:
        audio_file = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
        st.caption("Uses a free online speech-to-text service - best for short, clear recordings.")
        if not AUDIO_SUPPORT:
            st.warning("Audio transcription needs `speechrecognition` and `pydub` installed (see requirements.txt).")
        elif audio_file and st.button("Transcribe audio"):
            with st.spinner("Converting and transcribing audio... this can take a while."):
                wav_path = audio_file_to_wav(audio_file)
                st.session_state.current_transcript = transcribe_wav(wav_path)
            st.success("Transcription complete - review/edit below before analyzing.")

        st.text_area(
            "Transcript (from audio)",
            height=280,
            key="current_transcript",
        )

    transcript_text = st.session_state.current_transcript

    if st.button("🧠 Analyze Meeting", type="primary", disabled=not transcript_text.strip()):
        run_pipeline(transcript_text, meeting_title)
        st.success("Analysis complete! Head to the Dashboard tab.")

# ---------------------------------------------------------------- Dashboard
elif page == "📊 Dashboard":
    data = st.session_state.analysis
    if data:
        data = refresh_if_stale(data)
    if not data:
        st.info("No meeting analyzed yet. Go to **New Meeting** first, or load one from **History**.")
    else:
        st.header(f"📊 Dashboard — {data['title']}")

        st.subheader("📋 Meeting Summary")
        st.write(data["summary"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Key Discussion Points")
            for p in data["key_points"]:
                st.markdown(f"- {p}")
        with col2:
            st.subheader("🎯 Decisions Made")
            if data["decisions"]:
                for d in data["decisions"]:
                    st.markdown(f"- {d}")
            else:
                st.caption("No explicit decisions detected.")

        st.subheader("✅ Action Items")
        df = pd.DataFrame(data["action_items"])
        if not df.empty:
            edited = st.data_editor(
                df[["person", "task", "deadline", "priority", "status"]],
                column_config={
                    "status": st.column_config.SelectboxColumn(
                        options=["Pending", "In Progress", "Completed"]
                    ),
                    "priority": st.column_config.SelectboxColumn(
                        options=["High", "Medium", "Low"]
                    ),
                },
                num_rows="dynamic", use_container_width=True, key="editor",
            )
            data["action_items"] = edited.to_dict("records")
        else:
            st.caption("No action items detected.")

        st.subheader("🏷️ Important Entities")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("People", len(data["entities"]["people"]))
        e2.metric("Organizations", len(data["entities"]["organizations"]))
        e3.metric("Dates", len(data["entities"]["dates"]))
        e4.metric("Projects/Tech", len(data["entities"]["projects_tech"]))
        with st.expander("View entity details"):
            st.json(data["entities"])

        st.subheader("📈 Visual Insights")
        v1, v2 = st.columns(2)
        with v1:
            if data["action_items"]:
                counts = pd.Series([i["person"] for i in data["action_items"]]).value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind="barh", ax=ax, color="#4F46E5")
                ax.set_xlabel("Number of tasks")
                ax.set_title("Action Items by Person")
                st.pyplot(fig)
        with v2:
            if data["action_items"]:
                status_counts = pd.Series([i.get("status", "Pending") for i in data["action_items"]]).value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(status_counts, labels=status_counts.index, autopct="%1.0f%%",
                        colors=["#F59E0B", "#3B82F6", "#10B981"])
                ax2.set_title("Task Status")
                st.pyplot(fig2)

        if data["speaker_counts"] and len(data["speaker_counts"]) > 1:
            st.subheader("🎙️ Speaker Participation")
            sp_df = pd.Series(data["speaker_counts"]).sort_values(ascending=False)
            st.bar_chart(sp_df)

        st.subheader("📅 Upcoming Deadlines")
        deadline_df = pd.DataFrame(
            [i for i in data["action_items"] if i["deadline"] != "Not specified"]
        )
        if not deadline_df.empty:
            st.dataframe(deadline_df[["person", "task", "deadline"]], use_container_width=True)
        else:
            st.caption("No explicit deadlines detected.")

        st.subheader("📧 Generated Email Drafts")
        drafts = generate_all_drafts(data["action_items"], data["title"])
        if drafts:
            for d in drafts:
                with st.expander(f"✉️ To: {d['to']} — {d['subject']}"):
                    st.text(d["body"])
        else:
            if data["action_items"]:
                st.warning(
                    "Tasks were found, but none could be linked to a person, so no drafts were made. "
                    "Start each line of the transcript with the speaker's name, e.g. 'Priya: I will fix the bug by Friday.'"
                )
            else:
                st.caption("No action items detected, so there are no email drafts.")

# ---------------------------------------------------------------- History
elif page == "📜 History":
    st.header("📜 Meeting History")
    meetings = list_meetings()
    if not meetings:
        st.info("No meetings saved yet.")
    else:
        for m in meetings:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"**{m['title']}**  \n_{m['created_at']}_  \n{m['summary'][:150]}...")
                if c2.button("Open", key=f"open_{m['id']}"):
                    full = get_meeting(m["id"])
                    st.session_state.analysis = full["data"]
                    st.success(f"Loaded '{m['title']}'. Go to Dashboard.")
                if c3.button("Delete", key=f"del_{m['id']}"):
                    delete_meeting(m["id"])
                    st.rerun()

# ---------------------------------------------------------------- Export
elif page == "📤 Export":
    data = st.session_state.analysis
    if data:
        data = refresh_if_stale(data)
    if not data:
        st.info("Analyze a meeting first.")
    else:
        st.header("📤 Export")

        st.subheader("Downloadable Report")
        c1, c2 = st.columns(2)
        with c1:
            try:
                docx_bytes = report_to_docx_bytes(
                    data["title"], data["summary"], data["key_points"],
                    data["decisions"], data["action_items"], data["entities"],
                )
                st.download_button("⬇️ Download Report (DOCX)", docx_bytes,
                                    file_name=f"{data['title']}_report.docx")
            except Exception as e:
                st.error(f"Couldn't generate the DOCX report: {e}")
        with c2:
            try:
                pdf_bytes = report_to_pdf_bytes(
                    data["title"], data["summary"], data["key_points"],
                    data["decisions"], data["action_items"], data["entities"],
                )
                st.download_button("⬇️ Download Report (PDF)", pdf_bytes,
                                    file_name=f"{data['title']}_report.pdf")
            except Exception as e:
                st.error(f"Couldn't generate the PDF report: {e}")

        st.subheader("Export Tasks to Task-Management Systems")
        st.caption("CSV imports directly into Trello, Asana, Jira, ClickUp and Notion.")
        csv_bytes = action_items_to_csv_bytes(data["action_items"])
        st.download_button("⬇️ Tasks as CSV", csv_bytes, file_name=f"{data['title']}_tasks.csv")
