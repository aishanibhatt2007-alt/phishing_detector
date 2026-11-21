"""
Feature engineering for phishing email detection.

Combines:
  1. Raw text (subject + body) -> fed to a TF-IDF vectorizer in the ML pipeline.
  2. Hand-crafted heuristic/structural features -> numeric signals known to
     correlate with phishing (urgency language, URL shorteners, sender/domain
     mismatches, suspicious TLDs, etc.)

`extract_heuristic_features(email)` returns a dict of numeric features AND a
list of human-readable "reasons" describing which red flags fired, so the web
app can show *why* an email was flagged, not just a score.
"""
import re
from urllib.parse import urlparse

try:
    import tldextract
    # Disable remote fetching of the public suffix list (no network access in
    # sandboxed/offline environments) - use the bundled snapshot instead.
    _tldextractor = tldextract.TLDExtract(suffix_list_urls=())
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

URGENCY_WORDS = [
    "urgent", "immediately", "verify your account", "suspended", "act now",
    "final notice", "24 hours", "confirm your identity", "limited time",
    "click here", "password expires", "unusual activity", "security alert",
    "update your billing", "claim your refund", "failure to act",
    "will be closed", "will be suspended", "restricted", "unauthorized",
    "congratulations", "you have been selected", "winner", "free",
]

SENSITIVE_REQUEST_WORDS = [
    "ssn", "social security", "password", "credit card", "pin number",
    "bank account", "login credentials", "cvv", "routing number",
]

SUSPICIOUS_TLDS = {
    "info", "zip", "top", "xyz", "click", "work", "support", "country",
    "biz", "tk", "gq", "ml",
}

# Brands commonly impersonated; used to detect lookalike / unrelated domains
IMPERSONATED_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "bank", "chase",
    "wellsfargo", "netflix", "dhl", "fedex", "docusign", "office365",
    "irs", "linkedin",
]

URL_REGEX = re.compile(r"https?://[^\s\)\]>\"']+", re.IGNORECASE)
IP_URL_REGEX = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd"}


def _get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.split("@")[-1]  # strip userinfo if present
    except Exception:
        return ""


def _registered_domain(domain: str) -> str:
    if _HAS_TLDEXTRACT:
        ext = _tldextractor(domain)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    # crude fallback: last two labels
    parts = domain.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain


def extract_heuristic_features(email: dict) -> dict:
    """
    email: dict with keys 'sender', 'subject', 'body' (all strings; missing ok)
    Returns: dict of {feature_name: numeric_value, ..., '_reasons': [str, ...]}
    """
    sender = (email.get("sender") or "").strip().lower()
    subject = (email.get("subject") or "")
    body = (email.get("body") or "")
    full_text = f"{subject}\n{body}".lower()

    reasons = []
    feats = {}

    # --- Urgency / pressure language ---
    urgency_hits = [w for w in URGENCY_WORDS if w in full_text]
    feats["urgency_word_count"] = len(urgency_hits)
    if urgency_hits:
        reasons.append(
            f"Uses urgency/pressure language ({', '.join(urgency_hits[:4])})"
        )

    # --- Requests for sensitive info ---
    sensitive_hits = [w for w in SENSITIVE_REQUEST_WORDS if w in full_text]
    feats["sensitive_request_count"] = len(sensitive_hits)
    if sensitive_hits:
        reasons.append(f"Requests sensitive information ({', '.join(sensitive_hits)})")

    # --- URLs ---
    urls = URL_REGEX.findall(full_text)
    feats["url_count"] = len(urls)

    ip_urls = [u for u in urls if IP_URL_REGEX.match(u)]
    feats["has_ip_url"] = int(len(ip_urls) > 0)
    if ip_urls:
        reasons.append("Contains a link using a raw IP address instead of a domain")

    shortener_hits = []
    suspicious_tld_hits = []
    brand_domain_mismatch = []
    for u in urls:
        domain = _get_domain(u)
        reg_domain = _registered_domain(domain)
        tld = reg_domain.split(".")[-1] if "." in reg_domain else ""
        if any(s in domain for s in SHORTENER_DOMAINS):
            shortener_hits.append(domain)
        if tld in SUSPICIOUS_TLDS:
            suspicious_tld_hits.append(reg_domain)
        for brand in IMPERSONATED_BRANDS:
            if brand in domain and brand not in reg_domain.split(".")[0]:
                # brand name appears in subdomain/path trickery, not the real root domain
                brand_domain_mismatch.append((brand, reg_domain))

    feats["shortener_url_count"] = len(shortener_hits)
    if shortener_hits:
        reasons.append(f"Uses a URL shortener to hide the real destination ({shortener_hits[0]})")

    feats["suspicious_tld_count"] = len(suspicious_tld_hits)
    if suspicious_tld_hits:
        reasons.append(f"Link uses an uncommon/high-risk domain ending ({suspicious_tld_hits[0]})")

    feats["brand_mismatch_count"] = len(brand_domain_mismatch)
    if brand_domain_mismatch:
        brand, dom = brand_domain_mismatch[0]
        reasons.append(f"Mentions '{brand}' but links to an unrelated domain ({dom})")

    # --- Sender domain analysis ---
    sender_domain = sender.split("@")[-1] if "@" in sender else ""
    # Only flag ACTUAL character substitutions (digits/letters swapped in for
    # look-alikes), never the correctly spelled brand name itself.
    LOOKALIKE_PATTERNS = [
        r"payp[a4]1\b",       # paypa1, p4yp4l-style -> requires a trailing "1" not "l"
        r"g[o0]{2}g1e",       # go0gle/g00gle with substituted letters
        r"rnicrosoft",        # rn instead of m
        r"app1e",             # 1 instead of l
        r"0utlook",
        r"netf1ix",
    ]
    sender_has_lookalike = any(re.search(p, sender_domain) for p in LOOKALIKE_PATTERNS)
    # Explicit digit-substitution check for amazon/apple/google (avoid matching the
    # correctly-spelled brand, which would be a false positive)
    if re.search(r"amaz[0]n", sender_domain) or re.search(r"g[0]{1,2}gle", sender_domain):
        sender_has_lookalike = True
    feats["sender_has_digit_substitution"] = int(sender_has_lookalike)
    if feats["sender_has_digit_substitution"]:
        reasons.append(f"Sender domain uses look-alike spelling ({sender_domain})")

    sender_reg_domain = _registered_domain(sender_domain) if sender_domain else ""
    feats["sender_mentions_brand_but_wrong_domain"] = 0
    for brand in IMPERSONATED_BRANDS:
        if brand in sender_domain and not sender_reg_domain.startswith(brand):
            feats["sender_mentions_brand_but_wrong_domain"] = 1
            reasons.append(
                f"Sender claims to be '{brand}' but domain is '{sender_domain}', "
                "not the brand's real domain"
            )
            break

    # --- Generic greeting / mass-mail style ---
    feats["has_generic_greeting"] = int(
        bool(re.search(r"\b(dear customer|dear user|dear valued|dear account holder)\b", full_text))
    )
    if feats["has_generic_greeting"]:
        reasons.append("Uses a generic greeting instead of your name")

    # --- Excessive punctuation / ALL CAPS shouting ---
    caps_words = re.findall(r"\b[A-Z]{4,}\b", subject + " " + body)
    feats["shouting_word_count"] = len(caps_words)
    if len(caps_words) >= 2:
        reasons.append("Excessive capitalization (shouting) in subject/body")

    feats["exclamation_count"] = full_text.count("!")

    # --- Basic length features ---
    feats["body_length"] = len(body)
    feats["subject_length"] = len(subject)

    feats["_reasons"] = reasons
    return feats


def combined_text(email: dict) -> str:
    """Text blob used for TF-IDF: subject + body + sender (helps catch lookalike senders)."""
    return f"{email.get('subject','')} {email.get('body','')} {email.get('sender','')}"
