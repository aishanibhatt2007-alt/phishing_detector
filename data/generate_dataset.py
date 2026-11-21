"""
Generates a synthetic-but-realistic labeled email dataset for training/demo purposes.

Schema: sender, subject, body, label  (label: 1 = phishing, 0 = legitimate)

This is a STARTER dataset so the project runs end-to-end out of the box.
For production use, replace/augment with a real corpus, e.g.:
  - Kaggle "Phishing Email Dataset" (Nazario + enron + SpamAssassin merge)
  - https://monkey.org/~jose/phishing/  (Nazario phishing corpus)
  - Enron email dataset (legitimate mail) + SpamAssassin public corpus
Keep the same column names and everything downstream (features.py, train.py) works unchanged.
"""
import csv
import random

random.seed(42)

PHISH_SENDERS = [
    "security-alert@paypa1-verify.com",
    "support@amaz0n-billing.net",
    "no-reply@appleid-confirm.com",
    "accounts@bankofamerica-secure.net",
    "it-helpdesk@company-portal-update.com",
    "admin@microsoftonline-verify.com",
    "hr@payroll-update-portal.com",
    "notifications@netflix-account-verify.com",
    "service@dhl-redelivery.info",
    "billing@docusign-notify.net",
]

LEGIT_SENDERS = [
    "notifications@github.com",
    "no-reply@accounts.google.com",
    "receipts@amazon.com",
    "statements@chase.com",
    "team@slack.com",
    "hr@mycompany.com",
    "it-support@mycompany.com",
    "no-reply@linkedin.com",
    "newsletter@nytimes.com",
    "calendar-notification@zoom.us",
]

PHISH_SUBJECTS = [
    "Urgent: Your account will be suspended in 24 hours",
    "Action Required: Verify your identity immediately",
    "Security Alert - Unusual sign-in activity detected",
    "Your payment could not be processed - update now",
    "FINAL NOTICE: Your mailbox storage is full",
    "You have a pending delivery - confirm your address",
    "IT Department: Password expires today",
    "Invoice attached - payment overdue",
    "Congratulations! You've been selected for a refund",
    "Re: Your DocuSign document is waiting for signature",
]

LEGIT_SUBJECTS = [
    "Your weekly GitHub digest",
    "Your Amazon order has shipped",
    "Statement is ready to view",
    "Reminder: Team standup at 10am",
    "You have a new connection on LinkedIn",
    "Zoom meeting invitation: Q3 Planning",
    "Your receipt from Amazon.com",
    "Welcome to the team - onboarding checklist",
    "Monthly newsletter - November edition",
    "Your Google security checkup",
]

PHISH_BODY_TEMPLATES = [
    "Dear Customer,\n\nWe detected unusual activity on your account. Your account will be "
    "suspended within 24 hours unless you verify your identity immediately. "
    "Click here to confirm your details: http://{domain}/verify?id={rand}\n\n"
    "Failure to act now will result in permanent account closure.\n\nSecurity Team",

    "ATTENTION,\n\nYour payment of $499.99 could not be processed. To avoid service "
    "interruption, please update your billing information within 12 hours: "
    "http://{domain}/billing-update\n\nThank you,\nBilling Department",

    "Hello,\n\nThis is an urgent message from the IT Helpdesk. Your password expires "
    "today. To keep your account active, verify your credentials now at "
    "http://{domain}/reset-password or your access will be revoked.\n\nIT Support",

    "You have 1 new secure message regarding your account. Due to a recent security "
    "upgrade, we require you to confirm your information immediately: "
    "http://{domain}/secure-login. Act now to avoid limited functionality.",

    "Congratulations! You are eligible for a refund of $350.00. Click below to claim "
    "your refund before it expires in 24 hours: http://{domain}/claim-refund?ref={rand}",

    "Your package could not be delivered due to an incomplete address. Please confirm "
    "your delivery details within 24 hours or the package will be returned: "
    "http://{domain}/redelivery",
]

LEGIT_DOMAINS = [
    "github.com", "amazon.com", "chase.com", "slack.com", "linkedin.com",
    "zoom.us", "google.com", "nytimes.com", "docs.google.com", "drive.google.com",
]

# IMPORTANT: legitimate emails also routinely contain links (tracking pages,
# meeting joins, document views). Earlier drafts of this dataset only put
# URLs in the phishing templates, which taught the model a spurious
# "has a link = phishing" shortcut that false-flagged real Zoom/GitHub mail.
# These templates fix that by including normal links to legitimate domains.
LEGIT_BODY_TEMPLATES = [
    "Hi,\n\nHere's your weekly digest of activity on repositories you're watching. "
    "3 pull requests were merged and 2 issues were closed this week. "
    "View the full summary: https://github.com/notifications\n\nThe GitHub Team",

    "Hello,\n\nYour order #{rand} has shipped and is on its way! You can track your "
    "package anytime from your orders page: https://amazon.com/orders/track?id={rand}\n\n"
    "Estimated delivery: 2-3 business days.\n\nThanks for shopping with us.",

    "Hi team,\n\nJust a reminder that our standup is at 10am today in the usual "
    "conference room. Please come prepared with your updates from yesterday.\n\nThanks!",

    "Hi,\n\nYour monthly statement is now available to view in your online account: "
    "https://chase.com/statements/view\n\nNo action is required unless you have "
    "questions about a transaction.\n\nThank you.",

    "Hello,\n\nWelcome to the team! Attached is your onboarding checklist covering "
    "IT setup, benefits enrollment, and your first week schedule. Let us know if you "
    "have any questions.\n\nHR Team",

    "Hi,\n\nYou're invited to the Q3 Planning meeting on Zoom. The meeting will cover "
    "roadmap priorities for next quarter.\n\nJoin Zoom Meeting: https://zoom.us/j/{rand}"
    "\n\nBest,\nCalendar Team",

    "Hi,\n\nYou have a new connection request on LinkedIn. View their profile and "
    "accept the invitation here: https://linkedin.com/in/connections/{rand}\n\n"
    "The LinkedIn Team",

    "Hi,\n\nWe noticed a new sign-in to your Google Account on a Windows device. "
    "If this was you, no action is needed. Review your recent activity: "
    "https://myaccount.google.com/security\n\nThe Google Accounts Team",

    "Hi,\n\nSomeone shared a document with you: 'Q3 Budget Review'. "
    "Open it here: https://docs.google.com/document/d/{rand}/edit\n\nGoogle Docs",

    "Hi,\n\nHere's the November edition of our newsletter, featuring this month's "
    "top stories. Read online: https://nytimes.com/newsletter/{rand}\n\nThanks for "
    "subscribing.",

    "Hi,\n\nA new message was posted in #general that mentions you. "
    "View it in Slack: https://mycompany.slack.com/archives/general/p{rand}\n\n"
    "The Slack Team",
]

PHISH_DOMAINS = [
    "paypa1-verify.com", "amaz0n-billing.net", "appleid-confirm.com",
    "secure-bankupdate.net", "microsoftonline-verify.com", "docusign-notify.net",
]


def make_row(label):
    rand = random.randint(10000, 99999)
    if label == 1:
        sender = random.choice(PHISH_SENDERS)
        subject = random.choice(PHISH_SUBJECTS)
        body_t = random.choice(PHISH_BODY_TEMPLATES)
        domain = random.choice(PHISH_DOMAINS)
        body = body_t.format(domain=domain, rand=rand)
    else:
        sender = random.choice(LEGIT_SENDERS)
        subject = random.choice(LEGIT_SUBJECTS)
        body_t = random.choice(LEGIT_BODY_TEMPLATES)
        body = body_t.format(rand=rand)
    return sender, subject, body, label


def main(n_per_class=250, out_path="data/emails.csv"):
    rows = []
    for _ in range(n_per_class):
        rows.append(make_row(1))
        rows.append(make_row(0))
    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path} ({n_per_class} per class)")


if __name__ == "__main__":
    main()
