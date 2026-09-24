"""
DB engine + session helper.

Uses SQLite by default (zero setup, ships with the repo, perfect for a
same-day build and a live demo) but reads DATABASE_URL from the
environment so it drops straight into Postgres in production without any
code changes -- worth mentioning in an interview as the reason it's built
this way.
"""

import os
from sqlmodel import SQLModel, Session, create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cx_sentinel.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
