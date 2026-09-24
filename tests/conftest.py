"""
Makes the test suite runnable even in environments where the heavier deps
(langgraph, sqlmodel, chromadb) aren't installed -- e.g. a locked-down CI
sandbox with restricted network access. If the real package imports fine,
it's used as-is; only a genuinely missing package gets a minimal stub. On
Render / your own machine with `pip install -r requirements.txt`, none of
this fires and the real libraries run.
"""

import sys
import types


def _stub_langgraph():
    try:
        import langgraph.graph  # noqa
        return
    except ImportError:
        pass

    lg = types.ModuleType("langgraph")
    lg_graph = types.ModuleType("langgraph.graph")

    class _StubStateGraph:
        def __init__(self, *a, **k): pass
        def add_node(self, *a, **k): pass
        def add_edge(self, *a, **k): pass
        def add_conditional_edges(self, *a, **k): pass
        def set_entry_point(self, *a, **k): pass
        def compile(self, *a, **k): return None

    lg_graph.StateGraph = _StubStateGraph
    lg_graph.END = "END"
    sys.modules["langgraph"] = lg
    sys.modules["langgraph.graph"] = lg_graph


def _stub_sqlmodel():
    try:
        import sqlmodel  # noqa
        return
    except ImportError:
        pass

    sm = types.ModuleType("sqlmodel")

    def Field(*a, **k):
        return None

    class SQLModel:
        def __init_subclass__(cls, **kwargs):
            pass
        metadata = types.SimpleNamespace(create_all=lambda *a, **k: None)

    class Session:
        def __init__(self, *a, **k):
            pass

    sm.SQLModel = SQLModel
    sm.Field = Field
    sm.Session = Session
    sm.create_engine = lambda *a, **k: None
    sm.select = lambda *a, **k: None
    sys.modules["sqlmodel"] = sm


def _stub_chromadb():
    try:
        import chromadb  # noqa
        return
    except ImportError:
        pass

    cd = types.ModuleType("chromadb")

    class PersistentClient:
        def __init__(self, *a, **k):
            pass

    sub = types.ModuleType("chromadb.utils")
    embedfn = types.ModuleType("chromadb.utils.embedding_functions")

    class DefaultEmbeddingFunction:
        def __init__(self, *a, **k):
            pass

    embedfn.DefaultEmbeddingFunction = DefaultEmbeddingFunction
    sub.embedding_functions = embedfn
    cd.PersistentClient = PersistentClient
    sys.modules["chromadb"] = cd
    sys.modules["chromadb.utils"] = sub
    sys.modules["chromadb.utils.embedding_functions"] = embedfn


_stub_langgraph()
_stub_sqlmodel()
_stub_chromadb()
