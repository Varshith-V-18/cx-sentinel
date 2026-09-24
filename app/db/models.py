"""
Database models for CX Sentinel.

Three tables map directly onto the agent's own workflow, which is
deliberately the story you tell an interviewer:

- Ticket        -> every inbound customer message and what the agent
                    decided about it (sentiment / intent / urgency / action)
- ApprovalItem  -> the human-in-the-loop queue: anything the agent judged
                    too risky to execute on its own waits here
- ActionLog     -> an audit trail of every tool call the agent actually
                    executed, with its result, so every action is traceable
"""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Ticket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    channel: str = "review"          # review | email | chat | ticket
    customer_name: str = "Unknown"
    text: str

    # --- Analysis agent output ---
    sentiment: Optional[str] = None      # positive | neutral | negative
    sentiment_score: Optional[float] = None  # -1.0 .. 1.0
    intent: Optional[str] = None         # e.g. refund_request, bug_report, praise, question
    urgency: Optional[str] = None        # low | medium | high
    topics: Optional[str] = None         # comma-separated keywords/topics

    # --- Planner decision ---
    route: Optional[str] = None          # auto_resolve | escalate
    risk_flags: Optional[str] = None     # comma-separated guardrail flags, if any

    # --- Resolution ---
    status: str = "processing"           # processing | auto_resolved | pending_approval | approved | rejected | failed
    draft_response: Optional[str] = None
    kb_sources: Optional[str] = None     # which KB docs were retrieved (RAG citations)
    final_response: Optional[str] = None

    # --- Agent reasoning trace (for the UI + interview demo) ---
    trace: Optional[str] = None          # JSON string: list of {step, detail}


class ApprovalItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ticket_id: int = Field(foreign_key="ticket.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    reason: str                          # why the planner escalated this
    suggested_action: str                # what the agent proposes to do
    status: str = "pending"              # pending | approved | rejected
    resolved_at: Optional[datetime] = None
    resolver_note: Optional[str] = None


class ActionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ticket_id: int = Field(foreign_key="ticket.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    tool_name: str                       # send_reply | update_crm | escalate_to_human | request_refund
    input_summary: str
    result: str                          # success | failed
    detail: Optional[str] = None
