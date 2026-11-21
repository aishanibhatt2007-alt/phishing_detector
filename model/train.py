"""
Train the phishing detector on data/emails.csv and save it to model/phishing_detector.joblib

Usage:
    python -m model.train
"""
import sys
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.detector import PhishingDetector  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "phishing_detector.joblib")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.fillna("")
    emails = df[["sender", "subject", "body"]].to_dict(orient="records")
    labels = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training on {len(X_train)} emails, testing on {len(X_test)} emails...")
    detector = PhishingDetector()
    detector.fit(X_train, y_train)

    probs = detector.predict_proba(X_test)
    preds = (probs >= 0.5).astype(int)

    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")

    print("\n=== Top phishing-associated terms learned ===")
    for term, weight in detector.top_text_features(10):
        print(f"  {term:30s} weight={weight:.3f}")

    joblib.dump(detector, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
