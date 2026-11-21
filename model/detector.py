"""
PhishingDetector: wraps a TF-IDF text pipeline + heuristic numeric features
+ a scikit-learn classifier into one object that can be trained, saved,
loaded, and used for inference with a single .predict(email_dict) call.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model.features import extract_heuristic_features, combined_text

HEURISTIC_FEATURE_NAMES = [
    "urgency_word_count",
    "sensitive_request_count",
    "url_count",
    "has_ip_url",
    "shortener_url_count",
    "suspicious_tld_count",
    "brand_mismatch_count",
    "sender_has_digit_substitution",
    "sender_mentions_brand_but_wrong_domain",
    "has_generic_greeting",
    "shouting_word_count",
    "exclamation_count",
    "body_length",
    "subject_length",
]


class PhishingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def _heuristic_matrix(self, emails: list[dict]):
        rows = []
        for e in emails:
            feats = extract_heuristic_features(e)
            rows.append([feats[name] for name in HEURISTIC_FEATURE_NAMES])
        return np.array(rows, dtype=float)

    def _build_X(self, emails: list[dict], fit: bool):
        texts = [combined_text(e) for e in emails]
        heur = self._heuristic_matrix(emails)

        if fit:
            text_X = self.vectorizer.fit_transform(texts)
            heur_scaled = self.scaler.fit_transform(heur)
        else:
            text_X = self.vectorizer.transform(texts)
            heur_scaled = self.scaler.transform(heur)

        heur_sparse = csr_matrix(heur_scaled)
        return hstack([text_X, heur_sparse]).tocsr()

    def fit(self, emails: list[dict], labels: list[int]):
        X = self._build_X(emails, fit=True)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(self, emails: list[dict]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted/loaded yet.")
        X = self._build_X(emails, fit=False)
        return self.clf.predict_proba(X)[:, 1]  # probability of class "phishing"=1

    def predict_one(self, email: dict) -> dict:
        """Full prediction for a single email: score, label, and explanation."""
        proba = float(self.predict_proba([email])[0])
        heur_feats = extract_heuristic_features(email)
        reasons = heur_feats.pop("_reasons")

        if proba >= 0.75:
            risk_level = "High"
        elif proba >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "phishing_probability": round(proba, 4),
            "risk_level": risk_level,
            "is_phishing": bool(proba >= 0.5),
            "reasons": reasons,
            "heuristic_features": heur_feats,
        }

    def top_text_features(self, n=15):
        """Return the top-n TF-IDF terms most associated with phishing, for inspection."""
        if not self._fitted:
            return []
        coefs = self.clf.coef_[0][: len(self.vectorizer.get_feature_names_out())]
        terms = self.vectorizer.get_feature_names_out()
        order = np.argsort(coefs)[::-1][:n]
        return [(terms[i], float(coefs[i])) for i in order]
