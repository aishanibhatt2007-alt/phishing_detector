"""
Flask web app for the AI phishing detector.

Run with:
    python -m app.server

Then open http://localhost:5000
"""
import email
import email.utils
import os
import sys

from flask import Flask, render_template, request, jsonify
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.detector import PhishingDetector  # noqa: F401  (needed for joblib unpickling)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "phishing_detector.joblib"
)

app = Flask(__name__)
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "No trained model found. Run `python -m model.train` first."
            )
        _detector = joblib.load(MODEL_PATH)
    return _detector


def parse_eml_text(raw: str) -> dict:
    """Parse raw .eml / RFC822 text into sender/subject/body for the detector."""
    msg = email.message_from_string(raw)
    raw_from = msg.get("From", "")
    # Extract the clean email address from "Display Name <addr@domain>" format
    _, sender = email.utils.parseaddr(raw_from)
    sender = sender or raw_from
    subject = msg.get("Subject", "")

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    body = part.get_payload()
                break
        if not body:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        body = part.get_payload(decode=True).decode(errors="ignore")
                    except Exception:
                        body = part.get_payload()
                    break
    else:
        payload = msg.get_payload(decode=True)
        body = payload.decode(errors="ignore") if payload else str(msg.get_payload())

    return {"sender": sender, "subject": subject, "body": body}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}

    mode = data.get("mode", "fields")
    if mode == "raw":
        raw = data.get("raw_email", "")
        email_dict = parse_eml_text(raw)
    else:
        email_dict = {
            "sender": data.get("sender", ""),
            "subject": data.get("subject", ""),
            "body": data.get("body", ""),
        }

    if not any(email_dict.values()):
        return jsonify({"error": "Please provide email content to analyze."}), 400

    detector = get_detector()
    result = detector.predict_one(email_dict)
    result["parsed_email"] = email_dict
    return jsonify(result)


@app.route("/healthz")
def healthz():
    try:
        get_detector()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
