# AI-Powered Email Phishing Detector

A complete, working phishing-email detection system combining:

- **Machine learning** — a TF-IDF + Logistic Regression classifier trained on labeled email text
- **Heuristic rules** — hand-crafted checks for urgency language, look-alike sender domains, suspicious links (IP-based URLs, shorteners, risky TLDs), generic greetings, and more — these also power the human-readable explanations
- **A web app** (Flask) — paste an email or raw `.eml` source and get a risk score with reasons
- **A CLI batch scanner** — point it at a folder of `.eml` files for bulk triage/reporting

## Project layout

```
phishing-detector/
├── data/
│   ├── generate_dataset.py   # builds the starter labeled dataset
│   └── emails.csv            # generated training data (sender, subject, body, label)
├── model/
│   ├── features.py           # heuristic feature extraction + explanations
│   ├── detector.py           # PhishingDetector: TF-IDF + heuristics + classifier
│   ├── train.py              # trains and saves the model
│   └── phishing_detector.joblib  # trained model (generated)
├── app/
│   ├── server.py             # Flask app + /api/analyze endpoint
│   └── templates/index.html  # web UI
├── cli/
│   └── scan.py                # batch .eml scanner
├── sample_emails/             # example .eml files to try
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model

```bash
python -m model.train
```

This generates `model/phishing_detector.joblib`, trained on `data/emails.csv`
(500 synthetic-but-realistic emails, 250 phishing / 250 legitimate). It prints
a classification report, confusion matrix, and the top phishing-associated
terms the model learned.

To regenerate or expand the starter dataset:

```bash
python data/generate_dataset.py
```

**For production use**, swap in a real corpus — the code only needs a CSV
with `sender,subject,body,label` columns. Good public sources:
- Kaggle "Phishing Email Dataset" (merges Nazario + Enron + SpamAssassin)
- The Nazario phishing corpus
- Enron email dataset (legitimate) + SpamAssassin public corpus (spam/phishing)

## 2. Run the web app

```bash
python -m app.server
```

Open `http://localhost:5000`. You can:
- Paste a sender/subject/body directly
- Paste raw `.eml` / RFC822 source (headers + body)
- Try the built-in phishing/legitimate sample buttons

Each analysis returns a phishing probability, a High/Medium/Low risk level,
and a plain-English list of **why** (e.g. "Sender domain uses look-alike
spelling", "Uses urgency/pressure language", "Contains a link using a raw IP
address").

## 3. Batch-scan emails from the CLI

```bash
python -m cli.scan sample_emails/
python -m cli.scan path/to/inbox_export/ --json report.json --csv report.csv
python -m cli.scan one_email.eml
```

Prints a color-coded report per email plus a summary count, and can export
full JSON or a CSV summary for further triage.

## How detection works

1. **Text signal**: subject + body + sender are vectorized with TF-IDF
   (unigrams + bigrams) and fed to a Logistic Regression classifier.
2. **Heuristic signal**: ~14 numeric features are extracted per email —
   urgency-word count, sensitive-info requests, URL count, IP-based links,
   URL shorteners, suspicious TLDs, brand/domain mismatches, look-alike
   sender domains, generic greetings, shouting/exclamation density, etc.
   These are scaled and concatenated with the TF-IDF vector before
   classification, and are also surfaced directly as the "reasons" shown
   to the user — so the system isn't a black box.
3. The combined feature vector goes through a single `PhishingDetector`
   object (`model/detector.py`) that handles fit/predict/explain in one
   place, and is serialized with `joblib` for reuse by both the web app and
   the CLI.

## Notes & limitations

- The bundled dataset is synthetic (template-generated) so the project runs
  end-to-end with no external downloads. It's good for demonstrating the
  pipeline but a real deployment should retrain on a larger, real-world,
  continuously-updated corpus — phishing campaigns evolve quickly and a
  static model will drift.
- Heuristics (look-alike domains, brand names, suspicious TLDs) are
  intentionally conservative and illustrative; extend `model/features.py`
  with more rules (SPF/DKIM/DMARC header checks, attachment type checks,
  sending-infrastructure reputation, etc.) for production use.
- This is a decision-support tool, not a replacement for your organization's
  email security gateway / enterprise filtering.
