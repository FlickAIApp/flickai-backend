import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio"]
    style = request.form.get("style", "generic")

    if audio_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
    audio_file.save(temp_path)

    # ----- 1. TRANSCRIBE -----
    try:
        with open(temp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        transcript_text = transcription.text
    except Exception as e:
        return jsonify({"error": f"Transcription error: {str(e)}"}), 500

    # ----- 2. PROMPTS -----
    summary_prompts = {
        "client_meeting": """You are a note-taking assistant for a financial advisor at Iron Eagle Advisors.

Your job is to turn a phone call or meeting transcript into a clear, professional, compliance-friendly client meeting note.

Follow these rules:
- Write in the third person (e.g., “Christopher spoke with the Client…”).
- Use neutral, factual language.
- Do NOT invent any information that is not clearly supported in the transcript.
- If specific details (like date of birth, policy type, etc.) are mentioned, include them. If they are not mentioned, do not make them up.
- Keep the tone professional but readable, similar to an internal CRM note.

Format the output using the exact headings and structure below:

Discussion:
On [meeting date], [advisor or staff name], on behalf of [firm name], spoke with the Client, identified as [client name and any key identifiers if given], by [phone/in person/video]. Briefly summarize:
- Why the call/meeting occurred (incoming call, outreach, review, etc.).
- Any key context (e.g., client at work, limited time, follow-up from previous message).
- What was discussed at a high level (policy review, account info, service questions, etc.).

Then add 1–3 short paragraphs describing the main parts of the conversation in plain language, similar to a narrative case note.

Recommendations:
List any recommendations, suggestions, or actions the advisor or staff member indicated, using short bullet points.

Q + A:
If the client asked specific questions and received clear answers, list them in a Q/A format:

Q: [Client question in plain language]
A: [Advisor/staff answer in plain language]

Only include Q + A items that are clearly present in the transcript.

Outcomes and Next Steps:
Summarize concrete outcomes from the conversation:
- What was confirmed (e.g., identity, contact info, basic policy details).
- What was agreed to (e.g., schedule a review, send documents, follow-up call).
- How and when follow-up will occur, if mentioned.

Client Tasks:
List any actions the Client is expected to take (if any). If none are clearly mentioned, write: “None specifically discussed.”

Use concise sentences and keep the entire note suitable for documentation in a financial advisor’s CRM system.""",
        "email_recap": """You are an AI assistant writing a professional email recap to a client on behalf of a financial advisor at Iron Eagle Advisors.

Your goal is to take the transcript of a client call or meeting and produce a clear, concise, but friendly follow-up email that the advisor could send to the client.

Follow these rules:
- Write in a friendly but professional tone.
- Write from the advisor's perspective (“Hi [Client Name], here’s a quick recap of our conversation…”).
- If the client’s name or advisor’s name is mentioned in the transcript, include it. Do NOT invent names.
- Summarize only what was actually discussed.
- If a follow-up meeting was scheduled, clearly list the date and time.
- If the transcript does NOT give a date/time, simply write: “We will confirm a time that works best for you.”
- Include short bullet points for key discussion items.
- Include any next steps the advisor or client needs to take.
- Do NOT add recommendations or disclaimers that were not mentioned.
- Keep the email concise and easy to read.
-If a prompt in the structure doesn't exist delete
-If any section such as follow-up steps, next appointment, or recommendations does not apply or was not discussed, simply omit that section rather than inventing information or leaving it blank.
-When writing the email, keep the tone warm, conversational, and human — not robotic.
-Use natural phrasing that a real advisor or staff member would say in a follow-up message. Keep it professional, but allow small touches of friendliness.
-Generally maintain the provided structure, but allow the language and content to adjust naturally to reflect the real conversation. Ensure the summary remains professional and human in tone while staying within the boundaries of the defined format.

Avoid corporate jargon or overly formal language. Write the email as if you were genuinely emailing a client you’ve spoken with before.

Use the following structure:

Subject: Recap of Our Recent Conversation

Hi [Client Name],

Warm friendly two sentence greeting

Discussion Summary:
- …
- …
- …

Next Steps:
- …
- …

Follow-Up Appointment:
- If a next meeting was scheduled, list: “[Date] at [Time]”
- If not do not put anything.

Closing:
If you have any questions or need anything before our next conversation, feel free to reach out.

Best regards,
[Advisor Name]
Iron Eagle Advisors""",
        "generic": """You are a note-taking assistant for a financial advisor.
Summarize the following phone call in a structured, compliance-friendly way.

Format:

Purpose of Call:
- ...

Key Points Discussed:
- ...

Client Concerns:
- ...

Advisor Actions:
- ...

Follow-Up Items:
- ...

Use concise, neutral language. Do NOT invent facts."""
    }

    prompt_text = summary_prompts.get(style, summary_prompts["generic"])

    # ----- 3. SUMMARY -----
    try:
        summary = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{prompt_text}\n\nTranscript:\n{transcript_text}"
        )
        notes = summary.output_text
    except Exception as e:
        return jsonify({"error": f"Summary generation error: {str(e)}"}), 500

    os.remove(temp_path)

    return jsonify({
        "notes": notes,
        "transcript": transcript_text,
        "style": style
    })

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

if __name__ == "__main__":
    app.run(debug=True)

