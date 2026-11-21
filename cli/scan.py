"""
CLI batch scanner: scans a folder of .eml files (or a single .eml file) and
prints/report a phishing risk assessment for each one.

Usage:
    python -m cli.scan sample_emails/
    python -m cli.scan sample_emails/phish_example.eml
    python -m cli.scan sample_emails/ --json report.json
    python -m cli.scan sample_emails/ --csv report.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib  # noqa: E402
from model.detector import PhishingDetector  # noqa: F401,E402
from app.server import parse_eml_text  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "phishing_detector.joblib"
)

RESET = "\033[0m"
COLORS = {"High": "\033[91m", "Medium": "\033[93m", "Low": "\033[92m"}


def load_detector():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run `python -m model.train` first.", file=sys.stderr)
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def find_eml_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(".eml"):
                files.append(os.path.join(root, name))
    return sorted(files)


def scan_file(detector, filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    email_dict = parse_eml_text(raw)
    result = detector.predict_one(email_dict)
    result["file"] = filepath
    result["sender"] = email_dict.get("sender", "")
    result["subject"] = email_dict.get("subject", "")
    return result


def print_report(results):
    print(f"\nScanned {len(results)} email(s)\n" + "-" * 60)
    for r in results:
        color = COLORS.get(r["risk_level"], "")
        pct = round(r["phishing_probability"] * 100, 1)
        print(f"{color}[{r['risk_level']:>6}] {pct:>5}%{RESET}  {r['file']}")
        print(f"         From: {r['sender']}")
        print(f"         Subject: {r['subject']}")
        if r["reasons"]:
            for reason in r["reasons"]:
                print(f"           - {reason}")
        else:
            print("           - No specific red flags detected")
        print()

    n_high = sum(1 for r in results if r["risk_level"] == "High")
    n_med = sum(1 for r in results if r["risk_level"] == "Medium")
    n_low = sum(1 for r in results if r["risk_level"] == "Low")
    print("-" * 60)
    print(f"Summary: {n_high} High risk, {n_med} Medium risk, {n_low} Low risk")


def main():
    parser = argparse.ArgumentParser(description="Scan .eml files for phishing risk.")
    parser.add_argument("path", help="Path to a .eml file or a folder containing .eml files")
    parser.add_argument("--json", help="Write full results to this JSON file")
    parser.add_argument("--csv", help="Write a summary to this CSV file")
    args = parser.parse_args()

    files = find_eml_files(args.path)
    if not files:
        print(f"No .eml files found at {args.path}")
        sys.exit(1)

    detector = load_detector()
    results = [scan_file(detector, f) for f in files]

    print_report(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote detailed JSON report to {args.json}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "sender", "subject", "phishing_probability", "risk_level", "is_phishing"])
            for r in results:
                writer.writerow([r["file"], r["sender"], r["subject"],
                                  r["phishing_probability"], r["risk_level"], r["is_phishing"]])
        print(f"Wrote CSV summary to {args.csv}")


if __name__ == "__main__":
    main()
