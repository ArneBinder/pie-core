import dataclasses
from typing import Any, Dict, Optional

from pie_core import Annotation
from pie_core.document import (
    Document,
)


@dataclasses.dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def resolve(self) -> Any:
        return self.label, (self.head.resolve(), self.tail.resolve())


@dataclasses.dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(self.target[self.start : self.end])

    def resolve(self) -> Any:
        if self.is_attached:
            return self.target[self.start : self.end]
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclasses.dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclasses.dataclass
class TextBasedDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
