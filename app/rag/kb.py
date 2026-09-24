"""
RAG knowledge base over company policy documents, backed by ChromaDB.

Why RAG here (and not just "the LLM knows refund policy"): the agent's
draft replies must be *grounded* in the company's actual, current policy
text, not the model's guess at what a generic refund policy looks like.
Retrieval is also what lets the agent cite its sources in the trace --
"here is the policy doc I based this reply on" -- which is the difference
between an answer you can audit and one you can't.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")

_client = None
_collection = None

# A small, realistic policy knowledge base. In production this would be
# the company's actual support docs, ingested via a document pipeline;
# here it's hand-written so the demo is self-contained.
POLICY_DOCS = [
    {
        "id": "refund_policy",
        "title": "Refund Policy",
        "text": (
            "Customers can request a full refund within 30 days of purchase "
            "if the product is unused or defective. Refunds over $200 "
            "require manager approval before being processed. Digital "
            "products are non-refundable once downloaded, except in cases "
            "of confirmed billing error."
        ),
    },
    {
        "id": "shipping_policy",
        "title": "Shipping & Delivery Policy",
        "text": (
            "Standard shipping takes 5-7 business days. If an order is "
            "delayed beyond 10 business days, the customer is eligible for "
            "a shipping fee refund and a 10% discount coupon on their next "
            "order, applied automatically without manager approval."
        ),
    },
    {
        "id": "account_security",
        "title": "Account Security Policy",
        "text": (
            "Any report of suspected unauthorized account access, stolen "
            "payment information, or a compromised password must be "
            "escalated to a human security specialist immediately. The "
            "system must never attempt password resets or account changes "
            "automatically for security-related reports."
        ),
    },
    {
        "id": "billing_disputes",
        "title": "Billing Dispute Policy",
        "text": (
            "Billing disputes under $50 can be resolved by issuing a "
            "courtesy credit automatically. Disputes of $50 or more, or "
            "any dispute mentioning a chargeback or fraud claim, must be "
            "escalated to a human billing specialist."
        ),
    },
    {
        "id": "product_defect",
        "title": "Product Defect & Warranty Policy",
        "text": (
            "Products reported as defective within the warranty period "
            "(12 months) are eligible for a free replacement. The customer "
            "should be asked for a photo or description of the defect "
            "before a replacement ships."
        ),
    },
    {
        "id": "escalation_policy",
        "title": "General Escalation Policy",
        "text": (
            "Any message containing threats of legal action, mentions of "
            "self-harm, extremely abusive language, or explicit requests "
            "to speak to a manager must always be escalated to a human, "
            "regardless of the issue's dollar value."
        ),
    },
    {
        "id": "response_tone",
        "title": "Customer Response Tone Guidelines",
        "text": (
            "All automated replies must be empathetic, concise, and avoid "
            "over-promising. Never guarantee outcomes the company cannot "
            "control (e.g. exact delivery dates). Always thank the "
            "customer for their patience when there has been a delay."
        ),
    },
    {
        "id": "cancellation_policy",
        "title": "Subscription Cancellation Policy",
        "text": (
            "Customers can cancel a subscription at any time; the "
            "cancellation takes effect at the end of the current billing "
            "cycle and no partial refund is issued for the remaining days, "
            "unless the customer is a Premium tier subscriber, who is "
            "entitled to a prorated refund."
        ),
    },
]


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    _collection = _client.get_or_create_collection(
        name="cx_policy_kb", embedding_function=embed_fn
    )

    if _collection.count() == 0:
        _collection.add(
            ids=[d["id"] for d in POLICY_DOCS],
            documents=[d["text"] for d in POLICY_DOCS],
            metadatas=[{"title": d["title"]} for d in POLICY_DOCS],
        )
    return _collection


def search_policy(query: str, n_results: int = 2):
    """Return the top-N most relevant policy snippets for a query."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    for doc_id, meta, text in zip(ids, metas, docs):
        hits.append({"id": doc_id, "title": meta.get("title", doc_id), "text": text})
    return hits
