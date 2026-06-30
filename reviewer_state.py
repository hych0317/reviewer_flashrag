"""
Typed state contract for the ReviewerPipeline.

The pipeline still serializes plain dictionaries for FlashRAG compatibility, but
all EvidenceState dictionaries should pass through this module at state
boundaries. The schema is intentionally permissive: known fields are normalized,
unknown fields are preserved for older experiments and diagnostics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReviewDecision(str, Enum):
    CONTINUE_RETRIEVAL = "continue_retrieval"
    READY_TO_ANSWER = "ready_to_answer"
    FORCE_ANSWER = "force_answer"
    REVIEW_FAILED = "review_failed"

    @classmethod
    def coerce(cls, value: Any) -> "ReviewDecision":
        normalized = str(value or "").strip()
        legacy = {
            "NEED_MORE_INFO": cls.CONTINUE_RETRIEVAL.value,
            "READY_TO_ANSWER": cls.READY_TO_ANSWER.value,
        }
        normalized = legacy.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError:
            return cls.CONTINUE_RETRIEVAL


class SlotStatus(str, Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"

    @classmethod
    def coerce(cls, value: Any) -> "SlotStatus":
        try:
            return cls(str(value or "").strip())
        except ValueError:
            return cls.MISSING


class ClaimSupportStatus(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    IRRELEVANT = "irrelevant"

    @classmethod
    def coerce(cls, value: Any) -> "ClaimSupportStatus":
        try:
            return cls(str(value or "").strip())
        except ValueError:
            return cls.SUPPORT


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_str_list(value: Any) -> List[str]:
    return [str(item) for item in _as_list(value)]


def _as_int_list(value: Any) -> List[int]:
    items = []
    for item in _as_list(value):
        try:
            items.append(int(item))
        except (TypeError, ValueError):
            continue
    return items


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


@dataclass
class AnswerSlot:
    slot_id: str
    description: str
    slot_role: str
    depends_on: List[str] = field(default_factory=list)
    status: SlotStatus = SlotStatus.MISSING
    supporting_claim_ids: List[str] = field(default_factory=list)
    conflicting_claim_ids: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], idx: int = 0) -> "AnswerSlot":
        known = {
            "slot_id",
            "description",
            "slot_role",
            "depends_on",
            "status",
            "supporting_claim_ids",
            "conflicting_claim_ids",
        }
        return cls(
            slot_id=str(data.get("slot_id") or f"slot_{idx + 1}"),
            description=str(data.get("description") or f"Required evidence slot {idx + 1}"),
            slot_role=str(data.get("slot_role") or "direct_answer"),
            depends_on=_as_str_list(data.get("depends_on")),
            status=SlotStatus.coerce(data.get("status")),
            supporting_claim_ids=_as_str_list(data.get("supporting_claim_ids")),
            conflicting_claim_ids=_as_str_list(data.get("conflicting_claim_ids")),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        data.update({
            "slot_id": self.slot_id,
            "description": self.description,
            "slot_role": self.slot_role,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "supporting_claim_ids": self.supporting_claim_ids,
            "conflicting_claim_ids": self.conflicting_claim_ids,
        })
        return data


@dataclass
class EvidenceClaim:
    claim_id: str
    doc_id: str
    slot_id: str
    text: str
    support_status: ClaimSupportStatus = ClaimSupportStatus.SUPPORT
    round_id: Any = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], idx: int = 0) -> "EvidenceClaim":
        known = {"claim_id", "doc_id", "slot_id", "text", "support_status", "round_id"}
        return cls(
            claim_id=str(data.get("claim_id") or f"c{idx + 1}"),
            doc_id=str(data.get("doc_id") or ""),
            slot_id=str(data.get("slot_id") or ""),
            text=str(data.get("text") or ""),
            support_status=ClaimSupportStatus.coerce(data.get("support_status")),
            round_id=data.get("round_id", 0),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        data.update({
            "claim_id": self.claim_id,
            "doc_id": self.doc_id,
            "slot_id": self.slot_id,
            "text": self.text,
            "support_status": self.support_status.value,
            "round_id": self.round_id,
        })
        return data


@dataclass
class RetrievalPlanItem:
    slot_id: str
    query: str
    reason: str = "Need more evidence for this slot."
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalPlanItem":
        known = {"slot_id", "query", "reason"}
        return cls(
            slot_id=str(data.get("slot_id") or ""),
            query=str(data.get("query") or "").strip(),
            reason=str(data.get("reason") or "Need more evidence for this slot."),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        data.update({
            "slot_id": self.slot_id,
            "query": self.query,
            "reason": self.reason,
        })
        return data


@dataclass
class ReviewState:
    question_type: str = "single_hop"
    answer_slots: List[AnswerSlot] = field(default_factory=list)
    claims: List[EvidenceClaim] = field(default_factory=list)
    retrieval_plan: List[RetrievalPlanItem] = field(default_factory=list)
    decision: ReviewDecision = ReviewDecision.CONTINUE_RETRIEVAL
    answer_type: str = "phrase"
    direct_answer_supported: bool = False
    answer_candidate: Optional[str] = None
    answer_candidate_claim_ids: List[str] = field(default_factory=list)
    ready_guard_triggered: bool = False
    guard_extra_round_pending: bool = False
    guard_extra_round_used: bool = False
    docs_to_remove: List[int] = field(default_factory=list)
    ignored_docs: List[int] = field(default_factory=list)
    failure_reason: Optional[str] = None
    explanation: str = ""
    round_id: Any = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReviewState":
        data = data or {}
        known = {
            "question_type",
            "answer_slots",
            "claims",
            "retrieval_plan",
            "decision",
            "answer_type",
            "direct_answer_supported",
            "answer_candidate",
            "answer_candidate_claim_ids",
            "ready_guard_triggered",
            "guard_extra_round_pending",
            "guard_extra_round_used",
            "docs_to_remove",
            "ignored_docs",
            "failure_reason",
            "explanation",
            "round_id",
        }
        return cls(
            question_type=str(data.get("question_type") or "single_hop"),
            answer_slots=[
                AnswerSlot.from_dict(slot, idx)
                for idx, slot in enumerate(_as_list(data.get("answer_slots")))
                if isinstance(slot, dict)
            ],
            claims=[
                EvidenceClaim.from_dict(claim, idx)
                for idx, claim in enumerate(_as_list(data.get("claims")))
                if isinstance(claim, dict) and str(claim.get("text") or "").strip()
            ],
            retrieval_plan=[
                plan
                for plan in (
                    RetrievalPlanItem.from_dict(item)
                    for item in _as_list(data.get("retrieval_plan"))
                    if isinstance(item, dict)
                )
                if plan.query
            ],
            decision=ReviewDecision.coerce(data.get("decision")),
            answer_type=str(data.get("answer_type") or "phrase"),
            direct_answer_supported=_as_bool(data.get("direct_answer_supported")),
            answer_candidate=str(data.get("answer_candidate")).strip()
            if data.get("answer_candidate") is not None else None,
            answer_candidate_claim_ids=_as_str_list(data.get("answer_candidate_claim_ids")),
            ready_guard_triggered=_as_bool(data.get("ready_guard_triggered")),
            guard_extra_round_pending=_as_bool(data.get("guard_extra_round_pending")),
            guard_extra_round_used=_as_bool(data.get("guard_extra_round_used")),
            docs_to_remove=_as_int_list(data.get("docs_to_remove")),
            ignored_docs=_as_int_list(data.get("ignored_docs")),
            failure_reason=data.get("failure_reason"),
            explanation=str(data.get("explanation") or ""),
            round_id=data.get("round_id", 0),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        data.update({
            "question_type": self.question_type,
            "answer_slots": [slot.to_dict() for slot in self.answer_slots],
            "claims": [claim.to_dict() for claim in self.claims],
            "retrieval_plan": [plan.to_dict() for plan in self.retrieval_plan],
            "decision": self.decision.value,
            "answer_type": self.answer_type,
            "direct_answer_supported": self.direct_answer_supported,
            "answer_candidate": self.answer_candidate,
            "answer_candidate_claim_ids": self.answer_candidate_claim_ids,
            "ready_guard_triggered": self.ready_guard_triggered,
            "guard_extra_round_pending": self.guard_extra_round_pending,
            "guard_extra_round_used": self.guard_extra_round_used,
            "docs_to_remove": self.docs_to_remove,
            "ignored_docs": self.ignored_docs,
            "failure_reason": self.failure_reason,
            "explanation": self.explanation,
            "round_id": self.round_id,
        })
        return data

    def validate(self) -> List[str]:
        issues = []
        slot_ids = {slot.slot_id for slot in self.answer_slots}
        if not self.answer_slots:
            issues.append("answer_slots is empty")
        for claim in self.claims:
            if slot_ids and claim.slot_id not in slot_ids:
                issues.append(f"claim {claim.claim_id} references unknown slot {claim.slot_id}")
        for plan in self.retrieval_plan:
            if slot_ids and plan.slot_id not in slot_ids:
                issues.append(f"retrieval plan references unknown slot {plan.slot_id}")
        return issues

    def slot_coverage(self) -> Dict[str, Any]:
        total = len(self.answer_slots)
        supported = sum(1 for slot in self.answer_slots if slot.status == SlotStatus.SUPPORTED)
        conflicted = sum(1 for slot in self.answer_slots if slot.status == SlotStatus.CONFLICTED)
        missing = sum(
            1 for slot in self.answer_slots
            if slot.status in {SlotStatus.MISSING, SlotStatus.PARTIAL}
        )
        return {
            "total": total,
            "supported": supported,
            "conflicted": conflicted,
            "missing": missing,
            "coverage": supported / total if total else 0.0,
        }

    def compact_for_reviewer(self, claim_limit: int = 16) -> Dict[str, Any]:
        return {
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "answer_slots": [slot.to_dict() for slot in self.answer_slots],
            "claims": [claim.to_dict() for claim in self.claims[-claim_limit:]],
            "direct_answer_supported": self.direct_answer_supported,
            "answer_candidate": self.answer_candidate,
            "round_id": self.round_id,
        }


def normalize_review_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return ReviewState.from_dict(state).to_dict()
