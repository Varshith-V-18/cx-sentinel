from dotenv import load_dotenv
load_dotenv()

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db.database import init_db, get_session
from app.db.models import Ticket, ApprovalItem, ActionLog
from app.agent.graph import run_agent
from app.agent import tools, llm
from app.data.synthetic_tickets import SYNTHETIC_TICKETS

app = FastAPI(title="CX Sentinel", description="Agentic customer feedback triage & auto-resolution")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TicketIn(BaseModel):
    text: str
    channel: str = "review"
    customer_name: str = "Anonymous"


class ApprovalDecision(BaseModel):
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_ticket(session: Session, payload: TicketIn) -> Ticket:
    ticket = Ticket(
        channel=payload.channel,
        customer_name=payload.customer_name,
        text=payload.text,
        status="processing",
    )
    session.add(ticket)
    session.commit()
    session.refresh(ticket)

    final_state = run_agent(session, ticket.id, payload.text, payload.channel)

    ticket.sentiment = final_state.get("sentiment")
    ticket.sentiment_score = final_state.get("sentiment_score")
    ticket.intent = final_state.get("intent")
    ticket.urgency = final_state.get("urgency")
    ticket.topics = ", ".join(final_state.get("topics", []))
    ticket.route = final_state.get("route")
    ticket.risk_flags = ", ".join(final_state.get("risk_flags", []))
    ticket.status = final_state.get("status", "processing")
    ticket.draft_response = final_state.get("draft_response")
    ticket.final_response = final_state.get("final_response")
    ticket.kb_sources = json.dumps(final_state.get("kb_sources", []))
    ticket.trace = json.dumps(final_state.get("trace", []))

    session.add(ticket)
    session.commit()
    session.refresh(ticket)

    if ticket.status == "pending_approval":
        approval = ApprovalItem(
            ticket_id=ticket.id,
            reason=final_state.get("escalation_reason", "Needs review"),
            suggested_action=final_state.get("suggested_action", "Review manually"),
            status="pending",
        )
        session.add(approval)
        session.commit()

    return ticket


def _ticket_to_dict(t: Ticket) -> dict:
    return {
        "id": t.id,
        "created_at": t.created_at.isoformat(),
        "channel": t.channel,
        "customer_name": t.customer_name,
        "text": t.text,
        "sentiment": t.sentiment,
        "sentiment_score": t.sentiment_score,
        "intent": t.intent,
        "urgency": t.urgency,
        "topics": t.topics,
        "route": t.route,
        "risk_flags": t.risk_flags,
        "status": t.status,
        "draft_response": t.draft_response,
        "final_response": t.final_response,
        "kb_sources": json.loads(t.kb_sources) if t.kb_sources else [],
        "trace": json.loads(t.trace) if t.trace else [],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "llm_mode": "groq" if llm.llm_available() else "heuristic_fallback"}


@app.post("/api/seed")
def seed(session: Session = Depends(get_session)):
    existing = session.exec(select(Ticket)).all()
    if existing:
        return {"seeded": False, "reason": "tickets already exist", "count": len(existing)}

    created = []
    for row in SYNTHETIC_TICKETS:
        payload = TicketIn(text=row["text"], channel=row["channel"], customer_name=row["customer_name"])
        ticket = _process_ticket(session, payload)
        created.append(ticket.id)

    return {"seeded": True, "count": len(created)}


@app.post("/api/reset")
def reset(session: Session = Depends(get_session)):
    for model in (ActionLog, ApprovalItem, Ticket):
        for row in session.exec(select(model)).all():
            session.delete(row)
    session.commit()
    return {"ok": True}


@app.post("/api/tickets")
def create_ticket(payload: TicketIn, session: Session = Depends(get_session)):
    ticket = _process_ticket(session, payload)
    return _ticket_to_dict(ticket)


@app.get("/api/tickets")
def list_tickets(session: Session = Depends(get_session)):
    rows = session.exec(select(Ticket).order_by(Ticket.created_at.desc())).all()
    return [_ticket_to_dict(t) for t in rows]


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: int, session: Session = Depends(get_session)):
    t = session.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "not found")
    return _ticket_to_dict(t)


@app.get("/api/queue")
def get_queue(session: Session = Depends(get_session)):
    rows = session.exec(
        select(ApprovalItem).where(ApprovalItem.status == "pending").order_by(ApprovalItem.created_at)
    ).all()
    out = []
    for item in rows:
        ticket = session.get(Ticket, item.ticket_id)
        out.append({
            "id": item.id,
            "ticket_id": item.ticket_id,
            "reason": item.reason,
            "suggested_action": item.suggested_action,
            "created_at": item.created_at.isoformat(),
            "ticket": _ticket_to_dict(ticket) if ticket else None,
        })
    return out


@app.post("/api/queue/{item_id}/approve")
def approve_item(item_id: int, decision: ApprovalDecision, session: Session = Depends(get_session)):
    item = session.get(ApprovalItem, item_id)
    if not item or item.status != "pending":
        raise HTTPException(404, "pending approval item not found")

    ticket = session.get(Ticket, item.ticket_id)
    # Human approved -> now the agent is authorized to execute the
    # suggested action. This is the human-in-the-loop gate resolving.
    tools.send_reply(session, ticket.id, item.suggested_action)
    tools.update_crm(session, ticket.id, status="approved")

    item.status = "approved"
    item.resolved_at = datetime.utcnow()
    item.resolver_note = decision.note
    ticket.status = "approved"
    ticket.final_response = item.suggested_action

    trace = json.loads(ticket.trace) if ticket.trace else []
    trace.append({"step": "human_approval", "detail": f"Approved by user. Action executed: {item.suggested_action}"})
    ticket.trace = json.dumps(trace)

    session.add(item)
    session.add(ticket)
    session.commit()
    return {"ok": True}


@app.post("/api/queue/{item_id}/reject")
def reject_item(item_id: int, decision: ApprovalDecision, session: Session = Depends(get_session)):
    item = session.get(ApprovalItem, item_id)
    if not item or item.status != "pending":
        raise HTTPException(404, "pending approval item not found")

    ticket = session.get(Ticket, item.ticket_id)
    item.status = "rejected"
    item.resolved_at = datetime.utcnow()
    item.resolver_note = decision.note
    ticket.status = "rejected"

    trace = json.loads(ticket.trace) if ticket.trace else []
    trace.append({"step": "human_approval", "detail": f"Rejected by human reviewer. Note: {decision.note or '(none)'}"})
    ticket.trace = json.dumps(trace)

    session.add(item)
    session.add(ticket)
    session.commit()
    return {"ok": True}


@app.get("/api/analytics")
def analytics(session: Session = Depends(get_session)):
    rows = session.exec(select(Ticket)).all()
    total = len(rows)

    sentiment_counts = Counter(t.sentiment for t in rows if t.sentiment)
    status_counts = Counter(t.status for t in rows if t.status)
    intent_counts = Counter(t.intent for t in rows if t.intent)

    topic_counts = Counter()
    for t in rows:
        if t.topics:
            for topic in t.topics.split(","):
                topic = topic.strip()
                if topic:
                    topic_counts[topic] += 1

    by_day = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for t in rows:
        day = t.created_at.strftime("%Y-%m-%d")
        if t.sentiment in ("positive", "neutral", "negative"):
            by_day[day][t.sentiment] += 1

    route_counts = Counter(t.route for t in rows if t.route)
    auto_resolved = route_counts.get("auto_resolve", 0)
    escalated = route_counts.get("escalate", 0)

    return {
        "total": total,
        "sentiment_counts": dict(sentiment_counts),
        "status_counts": dict(status_counts),
        "intent_counts": dict(intent_counts),
        "top_topics": topic_counts.most_common(10),
        "sentiment_trend": [{"day": d, **v} for d, v in sorted(by_day.items())],
        "auto_resolution_rate": round(auto_resolved / total, 3) if total else 0,
        "escalation_rate": round(escalated / total, 3) if total else 0,
    }


# ---------------------------------------------------------------------------
# Serve dashboard (single-file frontend)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def dashboard():
    return FileResponse("app/static/index.html")
