"""
The agent's tools. Every one of these is (a) on the guardrails allow-list,
(b) logged to ActionLog for an audit trail, and (c) intentionally mocked
at the integration boundary (no real email/CRM system exists for a
same-day demo) -- but the boundary is exactly where a real send_email /
Zendesk / Salesforce API call would slot in, which is a fair thing to say
explicitly in an interview: "this is mocked at the integration edge, the
decision logic and audit trail around it are real."
"""

from sqlmodel import Session
from app.db.models import ActionLog
from app.security.guardrails import is_tool_allowed


def _log_action(session: Session, ticket_id: int, tool_name: str, input_summary: str,
                 result: str, detail: str = None):
    entry = ActionLog(
        ticket_id=ticket_id,
        tool_name=tool_name,
        input_summary=input_summary[:300],
        result=result,
        detail=detail,
    )
    session.add(entry)
    session.commit()
    return entry


def send_reply(session: Session, ticket_id: int, reply_text: str) -> dict:
    tool_name = "send_reply"
    if not is_tool_allowed(tool_name):
        raise PermissionError(f"Tool {tool_name} is not on the allow-list")
    # Mocked integration point: in production this calls the email/chat
    # provider's send API.
    _log_action(session, ticket_id, tool_name, reply_text, "success",
                detail="Mock send: no real email/chat provider wired up in demo.")
    return {"ok": True, "channel": "mock_email"}


def update_crm(session: Session, ticket_id: int, status: str) -> dict:
    tool_name = "update_crm"
    if not is_tool_allowed(tool_name):
        raise PermissionError(f"Tool {tool_name} is not on the allow-list")
    _log_action(session, ticket_id, tool_name, f"set status={status}", "success",
                detail="Mock CRM update: no real CRM wired up in demo.")
    return {"ok": True}


def log_credit(session: Session, ticket_id: int, amount_note: str) -> dict:
    tool_name = "log_credit"
    if not is_tool_allowed(tool_name):
        raise PermissionError(f"Tool {tool_name} is not on the allow-list")
    _log_action(session, ticket_id, tool_name, amount_note, "success",
                detail="Mock billing credit logged; real billing API call would go here.")
    return {"ok": True}


def escalate_to_human(session: Session, ticket_id: int, reason: str) -> dict:
    tool_name = "escalate_to_human"
    if not is_tool_allowed(tool_name):
        raise PermissionError(f"Tool {tool_name} is not on the allow-list")
    _log_action(session, ticket_id, tool_name, reason, "success",
                detail="Ticket added to human approval queue.")
    return {"ok": True}
