from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_handoff_email(
    *,
    subject: str,
    content: str,
    to_email: str,
    from_email: str,
    api_key: str,
) -> Optional[str]:
    if not (to_email and from_email and api_key):
        print("SendGrid: missing env vars (to/from/key)", flush=True)
        return None

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=content,
    )

    sg = SendGridAPIClient(api_key)
    resp = sg.send(message)
    print("SendGrid: status=", resp.status_code, flush=True)
    return str(resp.status_code)

