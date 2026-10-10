import os
import json
import re
import sqlite3
import smtplib
import ssl
import subprocess
from io import BytesIO
from email.message import EmailMessage
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from docx import Document
from pypdf import PdfReader


# ============================================================
# STEP 1 — ENVIRONMENT + GROQ
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key) if api_key else None

MODEL = "openai/gpt-oss-20b"
DATABASE_PATH = os.getenv(
    "MEETING_DB_PATH",
    os.getenv(
        "RECIPIENTS_DB_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipients.db")
    )
)


# ============================================================
# STEP 2 — STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Meeting Assistant")
st.caption("Multi-Agent Meeting Intelligence System")


# ============================================================
# SESSION MEMORY
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# GENERIC GROQ FUNCTION
# ============================================================

def ask_groq(system_prompt, user_prompt):

    if client is None:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file to run AI analysis.")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def read_uploaded_document(uploaded_file):

    file_name = uploaded_file.name
    extension = os.path.splitext(file_name)[1].lower()
    file_bytes = uploaded_file.getvalue()

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if extension == ".docx":
        document = Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    if extension in {".txt", ".md", ".csv", ".srt", ".vtt"}:
        return file_bytes.decode("utf-8-sig", errors="replace").strip()

    raise ValueError(f"Unsupported document format: {extension or 'unknown'}")


def transcribe_audio_file(file_name, file_bytes):

    if client is None:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file to transcribe audio.")

    if len(file_bytes) > 25 * 1024 * 1024:
        raise ValueError(f"{file_name} is larger than the 25 MB transcription limit.")

    try:
        response = client.audio.transcriptions.create(
            file=(file_name, file_bytes),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    except Exception as error:
        raise RuntimeError(f"Could not transcribe {file_name}: {error}") from error

    return response if isinstance(response, str) else str(response)


def transcribe_video_file(file_name, file_bytes):

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                "-f", "mp3", "pipe:1"
            ],
            input=file_bytes,
            capture_output=True,
            check=True,
            timeout=300
        )
    except FileNotFoundError as error:
        raise RuntimeError("Video transcription requires ffmpeg to be installed and on PATH.") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Could not extract audio from {file_name}: {details or 'invalid video file'}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Audio extraction timed out for {file_name}.") from error

    if not result.stdout:
        raise ValueError(f"No audio track was found in {file_name}.")

    return transcribe_audio_file(
        os.path.splitext(file_name)[0] + ".mp3",
        result.stdout
    )


def build_meeting_transcript(pasted_text, uploaded_files):

    transcript_parts = []
    if pasted_text.strip():
        transcript_parts.append(f"[Pasted meeting text]\n{pasted_text.strip()}")

    for uploaded_file in uploaded_files or []:
        file_name = uploaded_file.name
        extension = os.path.splitext(file_name)[1].lower()
        file_bytes = uploaded_file.getvalue()

        if extension in {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}:
            extracted = transcribe_audio_file(file_name, file_bytes)
        elif extension in {".mov", ".mkv", ".avi"}:
            extracted = transcribe_video_file(file_name, file_bytes)
        else:
            extracted = read_uploaded_document(uploaded_file)

        if extracted:
            transcript_parts.append(f"[From {file_name}]\n{extracted}")

    return "\n\n".join(transcript_parts)


# ============================================================
# STEP 3 — TRANSCRIPT ANALYZER
# ============================================================

def transcript_analyzer(transcript):

    system_prompt = """
You are a Transcript Analyzer Agent.

Analyze the meeting transcript.

Extract:
1. Participants
2. Important discussion points
3. Decisions
4. Tasks
5. Deadlines
6. Problems or blockers

Return clear structured information.
"""

    result = ask_groq(
        system_prompt,
        transcript
    )

    return result


# ============================================================
# STEP 4 — DECISION / ACTION AGENT
# ============================================================

def decision_agent(transcript_analysis):

    system_prompt = """
You are a Decision and Action Agent.

Read the meeting analysis.

Identify concrete actions.

For every task provide:

- Task
- Owner
- Deadline
- Priority

If owner or deadline is unknown, write "Not specified".
"""

    result = ask_groq(
        system_prompt,
        transcript_analysis
    )

    return result


# ============================================================
# STEP 5 — A2A WORKFLOW
# ============================================================

def a2a_workflow(transcript):

    # Agent 1
    analysis = transcript_analyzer(transcript)

    # Agent 2 receives Agent 1 output
    decisions = decision_agent(analysis)

    return {
        "transcript_analysis": analysis,
        "decisions": decisions
    }


# ============================================================
# STEP 6 — TASK ASSIGNMENT
# ============================================================

def task_assignment(decisions):

    system_prompt = """
You are a Task Assignment Agent.

Convert the meeting actions into a clean task list.

Use this format:

TASK 1
Owner:
Task:
Deadline:
Priority:

TASK 2
Owner:
Task:
Deadline:
Priority:

Use the participant's exact name from the meeting as the Owner.
Do not invent names. If no owner was assigned, write "Not specified".
Use the plain text labels exactly as shown, without Markdown formatting.
"""

    return ask_groq(
        system_prompt,
        decisions
    )


# ============================================================
# STEP 7 — EMAIL TOOL
# ============================================================

def email_tool(tasks):

    """
    Generate the follow-up email content that can be sent via SMTP.
    """

    system_prompt = """
You are an Email Agent.

Create a professional meeting follow-up email.

Include:
- Subject
- Greeting
- Assigned tasks
- Deadlines
- Closing

Do not invent information.
"""

    email = ask_groq(
        system_prompt,
        tasks
    )

    return email


# ============================================================
# STEP 7B — EMAIL RECIPIENT DATABASE + SENDING
# ============================================================

def initialize_recipient_database():

    os.makedirs(os.path.dirname(os.path.abspath(DATABASE_PATH)), exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recipients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                transcript TEXT NOT NULL,
                analysis TEXT NOT NULL,
                decisions TEXT NOT NULL,
                tasks TEXT NOT NULL,
                email_content TEXT NOT NULL,
                quality TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER,
                recipient_email TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                sent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE SET NULL
            )
            """
        )


def get_recipients():

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, name, email FROM recipients ORDER BY name COLLATE NOCASE"
        ).fetchall()

    return [dict(row) for row in rows]


def add_recipient(name, email):

    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Enter the recipient's name."

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                "INSERT INTO recipients (name, email) VALUES (?, ?)",
                (name, email)
            )
    except sqlite3.IntegrityError:
        return False, "That email address is already in the recipient database."

    return True, f"Added {name} to the recipient database."


def remove_recipient(recipient_id):

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM recipients WHERE id = ?", (recipient_id,))


def parse_assigned_tasks(tasks_text):

    task_blocks = re.split(
        r"(?im)^\s*(?:\*\*)?\s*TASK\s*\d+\s*:?\s*(?:\*\*)?\s*$",
        tasks_text
    )
    parsed_tasks = []

    for block in task_blocks:
        fields = {}
        for line in block.splitlines():
            match = re.match(
                r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*"
                r"(Owner|Task|Deadline|Priority)\s*:\s*"
                r"(.*?)(?:\*\*)?\s*$",
                line,
                flags=re.IGNORECASE
            )
            if match:
                fields[match.group(1).lower()] = match.group(2).strip()

        if fields.get("task"):
            parsed_tasks.append({
                "owner": fields.get("owner", "Not specified"),
                "task": fields["task"],
                "deadline": fields.get("deadline", "Not specified"),
                "priority": fields.get("priority", "Not specified")
            })

    return parsed_tasks


def normalize_person_name(name):

    return " ".join(re.findall(r"[a-z0-9]+", name.casefold()))


def tasks_for_saved_recipients(tasks_text, recipients):

    assignments = {}
    parsed_tasks = parse_assigned_tasks(tasks_text)

    for recipient in recipients:
        normalized_name = normalize_person_name(recipient["name"])
        normalized_email = normalize_person_name(recipient["email"])
        matched = []

        for task in parsed_tasks:
            owner = task["owner"].strip()
            owner_name = re.sub(r"\([^)]*\)", "", owner).strip()
            owner_names = [
                normalize_person_name(part)
                for part in re.split(r"\s*(?:,|/|&|\band\b)\s*", owner_name, flags=re.I)
            ]
            if not owner_names or any(
                owner_name in {"not specified", "unknown", "unassigned"}
                for owner_name in owner_names
            ):
                continue

            # Only exact saved-name/email matches are used; never guess from partial names.
            if normalized_name in owner_names or normalized_email in owner_names:
                matched.append(task)

        if matched:
            assignments[recipient["email"]] = matched

    return assignments


def build_assigned_work_email(recipient_name, assigned_tasks):

    lines = [
        f"Hello {recipient_name},",
        "",
        "Here is the work assigned to you in the meeting:",
        ""
    ]

    for index, task in enumerate(assigned_tasks, start=1):
        lines.extend([
            f"{index}. {task['task']}",
            f"   Deadline: {task['deadline']}",
            f"   Priority: {task['priority']}",
            ""
        ])

    lines.extend(["Please let the manager know if anything needs clarification.", ""])
    return "\n".join(lines)


def send_assigned_work_automatically(meeting_id, meeting_title, tasks_text):

    recipients = get_recipients()
    assignments = tasks_for_saved_recipients(tasks_text, recipients)
    recipient_by_email = {recipient["email"]: recipient for recipient in recipients}

    if not assignments:
        return {"sent": [], "failed": [], "unmatched": True}

    outgoing_messages = []
    subject = f"Work assigned: {meeting_title}"
    for address, assigned_tasks in assignments.items():
        recipient = recipient_by_email[address]
        outgoing_messages.append({
            "email": address,
            "name": recipient["name"],
            "subject": subject,
            "body": build_assigned_work_email(recipient["name"], assigned_tasks)
        })

    sent_addresses = []
    failed_addresses = []
    errors_by_address = {}
    try:
        sent, failed, errors_by_address = send_meeting_emails("", outgoing_messages)
        sent_addresses.extend(sent)
        failed_addresses.extend(failed)
        for address in sent:
            record_email_result(meeting_id, address, subject, "sent")
        for address in failed:
            record_email_result(
                meeting_id, address, subject, "failed",
                errors_by_address.get(address, "The SMTP server refused the recipient.")
            )
    except (ValueError, OSError, smtplib.SMTPException) as error:
        failed_addresses.extend(message["email"] for message in outgoing_messages)
        for message in outgoing_messages:
            record_email_result(
                meeting_id, message["email"], subject, "failed", str(error)
            )

    return {
        "sent": sent_addresses,
        "failed": failed_addresses,
        "unmatched": False
    }


def retry_failed_assigned_emails(meeting_id, meeting_title, tasks_text):

    recipients = get_recipients()
    assignments = tasks_for_saved_recipients(tasks_text, recipients)
    latest_by_address = {}
    for log in get_email_log(meeting_id):
        if log["recipient_email"] not in latest_by_address:
            latest_by_address[log["recipient_email"]] = log

    recipients_by_address = {recipient["email"]: recipient for recipient in recipients}
    retry_addresses = [
        address for address in assignments
        if latest_by_address.get(address, {}).get("status") == "failed"
    ]

    if not retry_addresses:
        return {"sent": [], "failed": [], "unmatched": False, "nothing_to_retry": True}

    subject = f"Work assigned: {meeting_title}"
    outgoing_messages = []
    for address in retry_addresses:
        recipient = recipients_by_address[address]
        outgoing_messages.append({
            "email": address,
            "name": recipient["name"],
            "subject": subject,
            "body": build_assigned_work_email(
                recipient["name"], assignments[address]
            )
        })

    try:
        sent, failed, errors_by_address = send_meeting_emails("", outgoing_messages)
        for address in sent:
            record_email_result(meeting_id, address, subject, "sent")
        for address in failed:
            record_email_result(
                meeting_id,
                address,
                subject,
                "failed",
                errors_by_address.get(address, "The SMTP server refused the recipient.")
            )
    except (ValueError, OSError, smtplib.SMTPException) as error:
        sent = []
        failed = retry_addresses
        for address in failed:
            record_email_result(meeting_id, address, subject, "failed", str(error))

    return {"sent": sent, "failed": failed, "unmatched": False, "nothing_to_retry": False}


def save_meeting(title, transcript, analysis, decisions, tasks, email_content,
                 quality, summary):

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO meetings (
                title, transcript, analysis, decisions, tasks,
                email_content, quality, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, transcript, analysis, decisions, tasks,
             email_content, quality, summary)
        )
        return cursor.lastrowid


def get_meetings():

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, title, created_at FROM meetings ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_meeting(meeting_id):

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_meeting(meeting_id):

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM email_log WHERE meeting_id = ?", (meeting_id,))
        cursor = connection.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    return cursor.rowcount > 0


def delete_all_meetings():

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM email_log")
        cursor = connection.execute("DELETE FROM meetings")
    return cursor.rowcount


def get_email_log(meeting_id=None):

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        if meeting_id is None:
            rows = connection.execute(
                """
                SELECT email_log.*, meetings.title AS meeting_title
                FROM email_log
                LEFT JOIN meetings ON meetings.id = email_log.meeting_id
                ORDER BY email_log.id DESC
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT email_log.*, meetings.title AS meeting_title
                FROM email_log
                LEFT JOIN meetings ON meetings.id = email_log.meeting_id
                WHERE email_log.meeting_id = ?
                ORDER BY email_log.id DESC
                """,
                (meeting_id,)
            ).fetchall()
    return [dict(row) for row in rows]


def record_email_result(meeting_id, recipient_email, subject, status, error=None):

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO email_log (
                meeting_id, recipient_email, subject, status, error
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (meeting_id, recipient_email, subject, status, error)
        )


def send_meeting_emails(email_content, recipients, subject_override=None):

    smtp_host = os.getenv("SMTP_HOST")
    smtp_username = os.getenv("SMTP_USERNAME")
    # Google displays App Passwords in groups separated by spaces.
    smtp_password = "".join((os.getenv("SMTP_PASSWORD") or "").split())
    sender = os.getenv("EMAIL_FROM") or smtp_username

    if smtp_host and smtp_host.casefold() == "smtp.gmail.com":
        # Gmail SMTP requires the From mailbox to be the authenticated account
        # (or a configured Gmail alias); default safely to the authenticated one.
        sender = smtp_username

    if not all((smtp_host, smtp_username, smtp_password, sender)):
        raise ValueError(
            "Email is not configured. Set SMTP_HOST, SMTP_USERNAME, "
            "SMTP_PASSWORD, and optionally EMAIL_FROM in your .env file."
        )

    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError as error:
        raise ValueError("SMTP_PORT must be a valid port number.") from error

    subject = subject_override or "Meeting follow-up"
    if subject_override is None:
        for line in email_content.splitlines():
            if line.strip().lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip() or subject
                break

    server_context = (
        smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        if smtp_port == 465
        else smtplib.SMTP(smtp_host, smtp_port, timeout=20)
    )

    sent = []
    failed = []
    errors_by_address = {}
    with server_context as server:
        if smtp_port != 465:
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        try:
            server.login(smtp_username, smtp_password)
        except smtplib.SMTPServerDisconnected as error:
            raise smtplib.SMTPException(
                "Gmail closed the SMTP login connection. Verify that "
                "SMTP_USERNAME is the Google account that generated the "
                "16-character App Password, and that SMTP_PASSWORD contains "
                "that App Password."
            ) from error

        for recipient in recipients:
            message = EmailMessage()
            message["From"] = sender
            message["To"] = recipient["email"]
            message["Subject"] = recipient.get("subject", subject)
            message.set_content(recipient.get("body", email_content))

            try:
                refused = server.send_message(message)
                if refused:
                    failed.append(recipient["email"])
                    errors_by_address[recipient["email"]] = str(refused)
                else:
                    sent.append(recipient["email"])
            except smtplib.SMTPException as error:
                failed.append(recipient["email"])
                errors_by_address[recipient["email"]] = str(error)

    return sent, failed, errors_by_address


initialize_recipient_database()


# ============================================================
# STEP 8 — MEMORY
# ============================================================

def save_memory(meeting_id, transcript, analysis, tasks):

    memory_item = {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "analysis": analysis,
        "tasks": tasks
    }

    st.session_state.memory.append(memory_item)


def get_memory():

    return st.session_state.memory


# ============================================================
# STEP 9 — MEETING QUALITY AGENT
# ============================================================

def meeting_quality_agent(transcript):

    system_prompt = """
You are a Meeting Quality Agent.

Evaluate the meeting.

Give scores from 1 to 10 for:

1. Clarity
2. Participation
3. Decision making
4. Actionability
5. Overall meeting quality

Also provide:

- Strengths
- Problems
- Suggestions for improvement
"""

    return ask_groq(
        system_prompt,
        transcript
    )


# ============================================================
# STEP 10 — SUMMARY AGENT
# ============================================================

def summary_agent(transcript):

    system_prompt = """
You are a Meeting Summary Agent.

Create a concise professional summary.

Include:

## Summary

## Key Decisions

## Important Discussion Points

## Action Items

## Deadlines

## Blockers
"""

    return ask_groq(
        system_prompt,
        transcript
    )


# ============================================================
# STEP 11 — AI MEETING CHAT
# ============================================================

def meeting_chat(question, transcript):

    system_prompt = """
You are an AI Meeting Assistant.

Answer questions using ONLY the meeting transcript.

If the answer cannot be found in the transcript,
say:

"I cannot find that information in the meeting."

Be concise and accurate.
"""

    return ask_groq(
        system_prompt,
        f"""
MEETING TRANSCRIPT:

{transcript}

USER QUESTION:

{question}
"""
    )


# ============================================================
# STEP 12 — FINAL UI
# ============================================================

st.sidebar.header("Meeting Input")

meeting_title = st.sidebar.text_input(
    "Meeting title",
    placeholder="e.g. Weekly project sync"
)

transcript_text = st.sidebar.text_area(
    "Paste meeting transcript or notes",
    height=400,
    placeholder="Paste meeting text, notes, or a transcript here..."
)

meeting_files = st.sidebar.file_uploader(
    "Add audio, video, or documents",
    type=[
        "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac",
        "mov", "mkv", "avi", "pdf", "docx", "txt", "md", "csv", "srt", "vtt"
    ],
    accept_multiple_files=True,
    help=(
        "Audio is transcribed with Groq Whisper. Video audio is extracted with ffmpeg. "
        "PDF, DOCX, TXT, Markdown, CSV, SRT, and VTT files are read as text. "
        "Audio files must be 25 MB or smaller each."
    )
)

st.sidebar.subheader("Recipients")
with st.sidebar.form("add_recipient_form", clear_on_submit=True):
    recipient_name_input = st.text_input("Recipient name")
    recipient_email_input = st.text_input("Recipient email")
    add_recipient_button = st.form_submit_button("Save recipient")

if add_recipient_button:
    was_added, recipient_message = add_recipient(
        recipient_name_input,
        recipient_email_input
    )
    if was_added:
        st.sidebar.success(recipient_message)
    else:
        st.sidebar.error(recipient_message)


analyze_button = st.sidebar.button(
    "🚀 Analyze Meeting",
    width="stretch"
)


# ============================================================
# RUN COMPLETE WORKFLOW
# ============================================================

if analyze_button:

    try:
        transcript = build_meeting_transcript(transcript_text, meeting_files)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
        transcript = ""

    if not api_key:

        st.error("GROQ_API_KEY is missing. Add it to your .env file to analyze meetings.")

    elif not transcript.strip():

        st.warning("Paste meeting text or add a supported audio, video, or document file.")

    else:

        with st.spinner("Running AI Meeting Agents..."):

            # ----------------------------------------
            # STEP 3 + STEP 4 + STEP 5
            # ----------------------------------------

            workflow = a2a_workflow(transcript)

            analysis = workflow["transcript_analysis"]

            decisions = workflow["decisions"]


            # ----------------------------------------
            # STEP 6
            # ----------------------------------------

            tasks = task_assignment(decisions)


            # ----------------------------------------
            # STEP 7
            # ----------------------------------------

            email = email_tool(tasks)


            # ----------------------------------------
            # STEP 9
            # ----------------------------------------

            quality = meeting_quality_agent(transcript)


            # ----------------------------------------
            # STEP 10
            # ----------------------------------------

            summary = summary_agent(transcript)

            saved_title = meeting_title.strip() or "Untitled meeting"
            meeting_id = save_meeting(
                saved_title,
                transcript,
                analysis,
                decisions,
                tasks,
                email,
                quality,
                summary
            )

            save_memory(meeting_id, transcript, analysis, tasks)

            delivery_result = send_assigned_work_automatically(
                meeting_id,
                saved_title,
                tasks
            )


            # Store results
            st.session_state.current_analysis = analysis
            st.session_state.current_decisions = decisions
            st.session_state.current_tasks = tasks
            st.session_state.current_email = email
            st.session_state.current_quality = quality
            st.session_state.current_summary = summary
            st.session_state.current_transcript = transcript
            st.session_state.current_meeting_id = meeting_id
            st.session_state.current_meeting_title = saved_title
            st.session_state.meeting_history_selector = meeting_id
            st.session_state.current_delivery_result = delivery_result


        st.success("Meeting analysis completed and saved.")
        if delivery_result["sent"]:
            st.success(
                "Assigned work was automatically emailed to: "
                + ", ".join(delivery_result["sent"])
            )
        if delivery_result["failed"]:
            st.error(
                "Automatic email delivery failed for: "
                + ", ".join(delivery_result["failed"])
                + ". See the Emails tab for the SMTP error details."
            )
        if delivery_result["unmatched"]:
            st.warning(
                "No saved recipient matched a task owner, so no email was sent. "
                "Make sure the task Owner name matches a saved recipient name exactly."
            )


# ============================================================
# DASHBOARD + PERSISTED MEETING WORKSPACE
# ============================================================

meeting_rows = get_meetings()
meeting_ids = [None] + [row["id"] for row in meeting_rows]
meeting_labels = {row["id"]: row for row in meeting_rows}

selected_meeting_id = st.sidebar.selectbox(
    "Open a saved meeting",
    options=meeting_ids,
    format_func=lambda meeting_id: (
        "Latest meeting" if meeting_id is None
        else f"{meeting_labels[meeting_id]['title']} — "
             f"{meeting_labels[meeting_id]['created_at']}"
    ),
    key="meeting_history_selector"
)

if selected_meeting_id is None and meeting_rows:
    selected_meeting_id = meeting_rows[0]["id"]

active_meeting = get_meeting(selected_meeting_id) if selected_meeting_id else None
all_email_logs = get_email_log()
sent_email_count = sum(log["status"] == "sent" for log in all_email_logs)
recipient_rows = get_recipients()

dashboard_tab, work_tab, decisions_tab, database_tab, email_tab, quality_tab, chat_tab = st.tabs(
    [
        "🏠 Dashboard",
        "📝 Work & Tasks",
        "✅ Decisions",
        "🗄️ Meeting Database",
        "📧 Emails",
        "📊 Quality",
        "💬 Meeting Chat"
    ]
)

with dashboard_tab:
    st.header("Meeting Dashboard")
    st.caption("Your meetings, action items, decisions, recipients, and delivery history in one place.")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Saved meetings", len(meeting_rows))
    metric2.metric("Saved recipients", len(recipient_rows))
    metric3.metric("Emails sent", sent_email_count)

    if active_meeting:
        st.subheader(active_meeting["title"])
        st.caption(f"Saved {active_meeting['created_at']} · Meeting #{active_meeting['id']}")
        st.markdown(active_meeting["summary"])
    else:
        st.info("No meeting saved yet. Enter a transcript in the sidebar and select Analyze Meeting.")

    st.subheader("Recent meetings")
    if meeting_rows:
        st.dataframe(
            [{"Title": row["title"], "Created": row["created_at"], "ID": row["id"]}
             for row in meeting_rows[:10]],
            width="stretch",
            hide_index=True
        )
    else:
        st.caption("Your meeting history will appear here after the first analysis.")

    with st.expander("🧠 Current-session meeting memory"):
        memory = get_memory()
        if not memory:
            st.write("No meeting memory in this session yet.")
        else:
            st.write(f"Meetings analyzed this session: {len(memory)}")
            for index, item in enumerate(memory, 1):
                st.markdown(f"### Meeting {index}")
                st.write(item["tasks"])

with work_tab:
    st.header("Work & Tasks")
    if active_meeting:
        st.subheader(active_meeting["title"])
        st.markdown(active_meeting["tasks"])
    else:
        st.info("Analyze a meeting to save and view its tasks here.")

with decisions_tab:
    st.header("Decisions & Meeting Analysis")
    if active_meeting:
        st.subheader("Decisions and actions")
        st.markdown(active_meeting["decisions"])
        with st.expander("Full transcript analysis"):
            st.markdown(active_meeting["analysis"])
    else:
        st.info("Saved meeting decisions will appear here.")

with database_tab:
    st.header("Meeting Database")
    st.caption(f"SQLite database: {os.path.basename(DATABASE_PATH)}")
    if meeting_rows:
        st.dataframe(
            [{"ID": row["id"], "Title": row["title"], "Created": row["created_at"]}
             for row in meeting_rows],
            width="stretch",
            hide_index=True
        )
        if active_meeting:
            st.subheader(f"Record: {active_meeting['title']}")
            with st.expander("Original transcript"):
                st.text(active_meeting["transcript"])
            st.download_button(
                "Download selected meeting as JSON",
                data=json.dumps(active_meeting, indent=2, ensure_ascii=False),
                file_name=f"meeting-{active_meeting['id']}.json",
                mime="application/json"
            )
    else:
        st.info("The SQLite meeting database is ready. No meeting records have been added yet.")

    st.subheader("Delete saved meeting data")
    st.caption(
        "Clearing Streamlit's cache does not delete records stored in this SQLite database. "
        "These controls remove meeting records and their related email delivery history; saved recipients are kept."
    )
    if active_meeting:
        confirm_delete_selected = st.checkbox(
            f"Confirm deletion of ‘{active_meeting['title']}’ and its email history",
            key=f"confirm_delete_selected_meeting_{active_meeting['id']}"
        )
        if st.button(
            "Delete selected meeting",
            disabled=not confirm_delete_selected,
            key="delete_selected_meeting"
        ):
            delete_meeting(active_meeting["id"])
            st.session_state.memory = [
                item for item in st.session_state.memory
                if (
                    item.get("meeting_id") != active_meeting["id"]
                    if item.get("meeting_id") is not None
                    else item.get("transcript") != active_meeting["transcript"]
                )
            ]
            st.success("Selected meeting and its email history were deleted.")
            st.rerun()

    if meeting_rows:
        confirm_delete_all = st.checkbox(
            "I understand this permanently deletes all saved meetings and email history",
            key="confirm_delete_all_meetings_"
                 + "-".join(str(row["id"]) for row in meeting_rows)
        )
        if st.button(
            "Delete all saved meetings",
            disabled=not confirm_delete_all,
            key="delete_all_meetings"
        ):
            delete_all_meetings()
            st.session_state.memory = []
            st.success("All saved meetings and their email history were deleted.")
            st.rerun()

    st.subheader("Saved email recipients")
    if recipient_rows:
        st.dataframe(
            [{"Name": row["name"], "Email": row["email"]}
             for row in recipient_rows],
            width="stretch",
            hide_index=True
        )
        recipient_ids = {
            f"{row['name']} <{row['email']}>": row["id"]
            for row in recipient_rows
        }
        recipient_to_remove = st.selectbox(
            "Remove a saved recipient",
            options=[None] + list(recipient_ids),
            format_func=lambda label: "Choose recipient" if label is None else label,
            key="recipient_to_remove"
        )
        if st.button("Remove recipient", key="remove_recipient_button"):
            if recipient_to_remove is None:
                st.warning("Choose a recipient first.")
            else:
                remove_recipient(recipient_ids[recipient_to_remove])
                st.success("Recipient removed from the database.")
                st.rerun()
    else:
        st.caption("Add recipients using the form in the sidebar.")

with email_tab:
    st.header("Email Follow-ups & Delivery History")
    retry_notice = st.session_state.pop("email_retry_notice", None)
    if retry_notice:
        if retry_notice["sent"]:
            st.success("Retry sent to: " + ", ".join(retry_notice["sent"]))
        if retry_notice["failed"]:
            st.error(
                "Retry still failed for: " + ", ".join(retry_notice["failed"])
            )
        if retry_notice.get("nothing_to_retry"):
            st.info("There are no failed assigned emails to retry for this meeting.")

    if active_meeting:
        st.subheader(f"Assigned work emails: {active_meeting['title']}")
        with st.expander("View full manager follow-up draft"):
            st.markdown(active_meeting["email_content"])

        assigned_tasks = tasks_for_saved_recipients(
            active_meeting["tasks"], recipient_rows
        )
        recipient_by_email = {row["email"]: row["name"] for row in recipient_rows}
        matched_addresses = list(assigned_tasks)

        if matched_addresses:
            st.caption(
                "Delivery runs automatically after meeting analysis. Only saved contacts "
                "whose names match task owners are emailed, and each receives only their tasks."
            )
            meeting_logs = get_email_log(active_meeting["id"])
            latest_status = {}
            for log in meeting_logs:
                if log["recipient_email"] not in latest_status:
                    latest_status[log["recipient_email"]] = log

            st.dataframe(
                [{"Recipient": recipient_by_email[address],
                  "Email": address,
                  "Assigned tasks": "\n".join(
                      task["task"] for task in assigned_tasks[address]
                  ),
                  "Delivery": latest_status.get(address, {}).get(
                      "status", "No attempt recorded"
                  ),
                  "Error": latest_status.get(address, {}).get("error") or ""}
                 for address in matched_addresses],
                width="stretch",
                hide_index=True
            )
            failed_addresses = [
                address for address in matched_addresses
                if latest_status.get(address, {}).get("status") == "failed"
            ]
            if failed_addresses:
                with st.expander("Why did delivery fail?"):
                    st.write(
                        "The recorded SMTP error was: "
                        + "; ".join(
                            f"{address}: "
                            f"{latest_status[address].get('error') or 'Unknown SMTP error'}"
                            for address in failed_addresses
                        )
                    )
                    st.markdown(
                        "For Gmail, enable 2-Step Verification and create a Gmail "
                        "**App Password** for the same account used as `SMTP_USERNAME`. "
                        "Put the 16-character App Password in `SMTP_PASSWORD` in your "
                        "local `.env` file (spaces are okay). Do not use your normal "
                        "Google account password. Then restart Streamlit and retry below."
                    )
                if st.button("Retry failed emails", key="retry_failed_emails"):
                    st.session_state.email_retry_notice = retry_failed_assigned_emails(
                        active_meeting["id"],
                        active_meeting["title"],
                        active_meeting["tasks"]
                    )
                    st.rerun()
        else:
            if not recipient_rows:
                st.info("Add people to the saved recipients database first.")
            else:
                st.warning(
                    "No saved recipient exactly matches an assigned task owner. "
                    "No email can be sent until task owners match saved recipient names."
                )
    else:
        st.info("Analyze or select a saved meeting to review assigned work email status.")

    st.subheader("Sent and failed email history")
    email_logs = get_email_log()
    if email_logs:
        st.dataframe(
            [{"Meeting": log["meeting_title"] or "Deleted meeting",
              "Recipient": log["recipient_email"], "Subject": log["subject"],
              "Status": log["status"], "Time": log["sent_at"],
              "Error": log["error"] or ""}
             for log in email_logs],
                        width="stretch",
            hide_index=True
        )
    else:
        st.caption("Delivery attempts for this meeting will be listed here.")

with quality_tab:
    st.header("Meeting Quality")
    if active_meeting:
        st.markdown(active_meeting["quality"])
    else:
        st.info("Meeting quality results will appear here after analysis.")

with chat_tab:
    st.header("Ask Your Meeting")
    if active_meeting:
        question = st.text_input("Ask something about the selected meeting", key="meeting_question")
        if st.button("Ask AI", key="ask_meeting_ai"):
            if question.strip():
                with st.spinner("Thinking..."):
                    answer = meeting_chat(question, active_meeting["transcript"])
                st.markdown("### AI Answer")
                st.write(answer)
    else:
        st.info("Select or analyze a meeting to ask questions about it.")

