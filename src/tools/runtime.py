"""Local runtime context used by LangGraph nodes.

The workflow only needs a request id.  Keeping this tiny context object avoids
pulling the Coze runtime into otherwise platform-independent business nodes.
"""
from dataclasses import dataclass, field
import uuid


@dataclass
class Context:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

