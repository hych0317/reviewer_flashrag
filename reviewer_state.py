"""
Typed state contract for the ReviewerPipeline.

The pipeline still serializes plain dictionaries for FlashRAG compatibility, but
all EvidenceState dictionaries should pass through this module at state
boundaries. The schema is intentionally permissive: known fields are normalized,
unknown fields are preserved for older experiments and diagnostics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


PIPELINE_VERSION = "ver2_7_patch9"
STATE_SCHEMA_VERSION = "reviewer_state_v2_7"
PROMPT_VERSION = "reviewer_prompt_v2_7_patch8"


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
        normalized = str(value or "").strip()
        legacy = {
            "contradicted": cls.CONTRADICT.value,
            "contradiction": cls.CONTRADICT.value,
            "conflict": cls.CONTRADICT.value,
            "conflicted": cls.CONTRADICT.value,
        }
        normalized = legacy.get(normalized, normalized)
        try:
            return cls(normalized)
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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AnswerSlot:
    slot_id: str
    description: str
    slot_role: str
    depends_on: List[str] = field(default_factory=list)
    status: SlotStatus = SlotStatus.MISSING
    supporting_claim_ids: List[str] = field(default_factory=list)
    conflicting_claim_ids: List[str] = field(default_factory=list)
    target_relation: Optional[str] = None
    relation_family: Optional[str] = None
    expected_answer_type: Optional[str] = None
    comparison_option: Optional[str] = None
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
            "target_relation",
            "relation_family",
            "expected_answer_type",
            "comparison_option",
        }
        return cls(
            slot_id=str(data.get("slot_id") or f"slot_{idx + 1}"),
            description=str(data.get("description") or f"Required evidence slot {idx + 1}"),
            slot_role=str(data.get("slot_role") or "direct_answer"),
            depends_on=_as_str_list(data.get("depends_on")),
            status=SlotStatus.coerce(data.get("status")),
            supporting_claim_ids=_as_str_list(data.get("supporting_claim_ids")),
            conflicting_claim_ids=_as_str_list(data.get("conflicting_claim_ids")),
            target_relation=data.get("target_relation"),
            relation_family=data.get("relation_family"),
            expected_answer_type=data.get("expected_answer_type"),
            comparison_option=data.get("comparison_option"),
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
            "target_relation": self.target_relation,
            "relation_family": self.relation_family,
            "expected_answer_type": self.expected_answer_type,
            "comparison_option": self.comparison_option,
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
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    object_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], idx: int = 0) -> "EvidenceClaim":
        known = {
            "claim_id", "doc_id", "slot_id", "text", "support_status", "round_id",
            "subject", "predicate", "object", "object_type",
        }
        return cls(
            claim_id=str(data.get("claim_id") or f"c{idx + 1}"),
            doc_id=str(data.get("doc_id") or ""),
            slot_id=str(data.get("slot_id") or ""),
            text=str(data.get("text") or ""),
            support_status=ClaimSupportStatus.coerce(data.get("support_status")),
            round_id=data.get("round_id", 0),
            subject=data.get("subject"),
            predicate=data.get("predicate"),
            object=data.get("object"),
            object_type=data.get("object_type"),
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
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
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
    pipeline_version: str = PIPELINE_VERSION
    state_schema_version: str = STATE_SCHEMA_VERSION
    question_type: str = "single_hop"
    answer_slots: List[AnswerSlot] = field(default_factory=list)
    claims: List[EvidenceClaim] = field(default_factory=list)
    retrieval_plan: List[RetrievalPlanItem] = field(default_factory=list)
    query_history: List[str] = field(default_factory=list)
    last_retrieval_plan: List[RetrievalPlanItem] = field(default_factory=list)
    last_retrieval_new_doc_count: int = 0
    no_new_claim_streak: int = 0
    recovery_triggered: bool = False
    recovery_count: int = 0
    recovery_reason: Optional[str] = None
    recovery_extra_round_pending: bool = False
    recovery_extra_round_used: bool = False
    decision: ReviewDecision = ReviewDecision.CONTINUE_RETRIEVAL
    answer_type: str = "phrase"
    direct_answer_supported: bool = False
    answer_candidate: Optional[str] = None
    answer_candidate_claim_ids: List[str] = field(default_factory=list)
    answer_intent: Dict[str, Any] = field(default_factory=dict)
    comparison_output_contract: Dict[str, Any] = field(default_factory=dict)
    relation_grounding_trace: Dict[str, Any] = field(default_factory=dict)
    answer_decision: Dict[str, Any] = field(default_factory=dict)
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
            "pipeline_version",
            "state_schema_version",
            "question_type",
            "answer_slots",
            "claims",
            "retrieval_plan",
            "query_history",
            "last_retrieval_plan",
            "last_retrieval_new_doc_count",
            "no_new_claim_streak",
            "recovery_triggered",
            "recovery_count",
            "recovery_reason",
            "recovery_extra_round_pending",
            "recovery_extra_round_used",
            "decision",
            "answer_type",
            "direct_answer_supported",
            "answer_candidate",
            "answer_candidate_claim_ids",
            "answer_intent",
            "comparison_output_contract",
            "relation_grounding_trace",
            "answer_decision",
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
            pipeline_version=str(data.get("pipeline_version") or PIPELINE_VERSION),
            state_schema_version=str(data.get("state_schema_version") or STATE_SCHEMA_VERSION),
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
            query_history=_as_str_list(data.get("query_history")),
            last_retrieval_plan=[
                plan
                for plan in (
                    RetrievalPlanItem.from_dict(item)
                    for item in _as_list(data.get("last_retrieval_plan"))
                    if isinstance(item, dict)
                )
                if plan.query
            ],
            last_retrieval_new_doc_count=_as_int(data.get("last_retrieval_new_doc_count")),
            no_new_claim_streak=_as_int(data.get("no_new_claim_streak")),
            recovery_triggered=_as_bool(data.get("recovery_triggered")),
            recovery_count=_as_int(data.get("recovery_count")),
            recovery_reason=data.get("recovery_reason"),
            recovery_extra_round_pending=_as_bool(data.get("recovery_extra_round_pending")),
            recovery_extra_round_used=_as_bool(data.get("recovery_extra_round_used")),
            decision=ReviewDecision.coerce(data.get("decision")),
            answer_type=str(data.get("answer_type") or "phrase"),
            direct_answer_supported=_as_bool(data.get("direct_answer_supported")),
            answer_candidate=str(data.get("answer_candidate")).strip()
            if data.get("answer_candidate") is not None else None,
            answer_candidate_claim_ids=_as_str_list(data.get("answer_candidate_claim_ids")),
            answer_intent=dict(data.get("answer_intent") or {}),
            comparison_output_contract=dict(data.get("comparison_output_contract") or {}),
            relation_grounding_trace=dict(data.get("relation_grounding_trace") or {}),
            answer_decision=dict(data.get("answer_decision") or {}),
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
        schema_extra_fields = sorted(key for key in data if key != "schema_extra_fields")
        data.update({
            "pipeline_version": self.pipeline_version or PIPELINE_VERSION,
            "state_schema_version": self.state_schema_version or STATE_SCHEMA_VERSION,
            "question_type": self.question_type,
            "answer_slots": [slot.to_dict() for slot in self.answer_slots],
            "claims": [claim.to_dict() for claim in self.claims],
            "retrieval_plan": [plan.to_dict() for plan in self.retrieval_plan],
            "query_history": self.query_history,
            "last_retrieval_plan": [plan.to_dict() for plan in self.last_retrieval_plan],
            "last_retrieval_new_doc_count": self.last_retrieval_new_doc_count,
            "no_new_claim_streak": self.no_new_claim_streak,
            "recovery_triggered": self.recovery_triggered,
            "recovery_count": self.recovery_count,
            "recovery_reason": self.recovery_reason,
            "recovery_extra_round_pending": self.recovery_extra_round_pending,
            "recovery_extra_round_used": self.recovery_extra_round_used,
            "decision": self.decision.value,
            "answer_type": self.answer_type,
            "direct_answer_supported": self.direct_answer_supported,
            "answer_candidate": self.answer_candidate,
            "answer_candidate_claim_ids": self.answer_candidate_claim_ids,
            "answer_intent": self.answer_intent,
            "comparison_output_contract": self.comparison_output_contract,
            "relation_grounding_trace": self.relation_grounding_trace,
            "answer_decision": self.answer_decision,
            "ready_guard_triggered": self.ready_guard_triggered,
            "guard_extra_round_pending": self.guard_extra_round_pending,
            "guard_extra_round_used": self.guard_extra_round_used,
            "docs_to_remove": self.docs_to_remove,
            "ignored_docs": self.ignored_docs,
            "failure_reason": self.failure_reason,
            "explanation": self.explanation,
            "round_id": self.round_id,
            "schema_extra_fields": schema_extra_fields,
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
        evidence_slots = self.answer_slots
        if self.question_type == "comparison":
            evidence_slots = [
                slot for slot in self.answer_slots
                if slot.slot_role != "final_answer" and not slot.extra.get("derived_output")
            ]
        total = len(evidence_slots)
        supported = sum(1 for slot in evidence_slots if slot.status == SlotStatus.SUPPORTED)
        conflicted = sum(1 for slot in evidence_slots if slot.status == SlotStatus.CONFLICTED)
        missing = sum(
            1 for slot in evidence_slots
            if slot.status in {SlotStatus.MISSING, SlotStatus.PARTIAL}
        )
        operand_slots = [slot for slot in self.answer_slots if slot.slot_role == "comparison_operand"]
        operand_supported = sum(1 for slot in operand_slots if slot.status == SlotStatus.SUPPORTED)
        operand_conflicted = sum(1 for slot in operand_slots if slot.status == SlotStatus.CONFLICTED)
        operand_missing = sum(
            1 for slot in operand_slots
            if slot.status in {SlotStatus.MISSING, SlotStatus.PARTIAL}
        )
        operand_total = len(operand_slots)
        return {
            "total": total,
            "supported": supported,
            "conflicted": conflicted,
            "missing": missing,
            "coverage": supported / total if total else 0.0,
            "evidence_slot_coverage": supported / total if total else 0.0,
            "comparison_operand_coverage": operand_supported / operand_total if operand_total else 0.0,
            "comparison_operand_total": operand_total,
            "comparison_operand_supported": operand_supported,
            "comparison_operand_conflicted": operand_conflicted,
            "comparison_operand_missing": operand_missing,
        }

    def compact_for_reviewer(self, claim_limit: int = 16) -> Dict[str, Any]:
        return {
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "answer_slots": [slot.to_dict() for slot in self.answer_slots],
            "claims": [claim.to_dict() for claim in self.claims[-claim_limit:]],
            "direct_answer_supported": self.direct_answer_supported,
            "answer_candidate": self.answer_candidate,
            "answer_intent": self.answer_intent,
            "round_id": self.round_id,
        }


def normalize_review_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return ReviewState.from_dict(state).to_dict()


def repair_slot_claim_consistency(
    state: Optional[Dict[str, Any]],
    question: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Repair deterministic slot status drift introduced by LLM slot updates.

    The reviewer can emit support claims for a slot while leaving that slot as
    missing. This helper makes the slot view consistent with the claim view
    without changing the schema contract.
    """
    normalized = ReviewState.from_dict(state).to_dict()
    support_by_slot: Dict[str, List[Dict[str, Any]]] = {}
    conflict_by_slot: Dict[str, List[Dict[str, Any]]] = {}
    for claim in normalized.get("claims") or []:
        slot_id = str(claim.get("slot_id") or "")
        claim_id = str(claim.get("claim_id") or "")
        if not slot_id or not claim_id:
            continue
        status = str(claim.get("support_status") or "")
        if status == ClaimSupportStatus.SUPPORT.value:
            support_by_slot.setdefault(slot_id, []).append(claim)
        elif status == ClaimSupportStatus.CONTRADICT.value:
            conflict_by_slot.setdefault(slot_id, []).append(claim)

    issues: List[Dict[str, Any]] = []
    for slot in normalized.get("answer_slots") or []:
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        old_status = str(slot.get("status") or SlotStatus.MISSING.value)
        support_claims = support_by_slot.get(slot_id, [])
        conflict_claims = conflict_by_slot.get(slot_id, [])
        support_ids = _dedupe_strs(
            list(slot.get("supporting_claim_ids") or [])
            + [claim.get("claim_id") for claim in support_claims]
        )
        conflict_ids = _dedupe_strs(
            list(slot.get("conflicting_claim_ids") or [])
            + [claim.get("claim_id") for claim in conflict_claims]
        )
        new_status = old_status
        reason = None
        blocked_reason = None
        observed = (
            bool(support_ids and old_status in {SlotStatus.MISSING.value, SlotStatus.PARTIAL.value})
            or bool(support_ids and conflict_ids)
        )
        repair_allowed = _slot_repair_allowed(slot, support_claims, question)
        if support_ids and conflict_ids:
            new_status = SlotStatus.CONFLICTED.value
            reason = "support_and_contradict_claims"
        elif support_ids and old_status in {SlotStatus.MISSING.value, SlotStatus.PARTIAL.value}:
            if repair_allowed:
                new_status = SlotStatus.SUPPORTED.value
                reason = "missing_slot_has_support_claim"
            else:
                blocked_reason = _slot_repair_blocked_reason(slot, support_claims, question)
        if reason or observed:
            issues.append({
                "slot_id": slot_id,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason or "observed_support_claim",
                "observed": observed,
                "repaired": bool(reason),
                "blocked_reason": blocked_reason,
                "supporting_claim_ids": support_ids,
                "conflicting_claim_ids": conflict_ids,
            })
        slot["status"] = new_status
        slot["supporting_claim_ids"] = support_ids
        slot["conflicting_claim_ids"] = conflict_ids
    return normalized, issues


def _slot_repair_allowed(
    slot: Dict[str, Any],
    support_claims: List[Dict[str, Any]],
    question: Optional[str],
) -> bool:
    role = str(slot.get("slot_role") or "")
    if role not in {"final_answer", "direct_answer"}:
        return False
    slot_text = " ".join([
        str(slot.get("description") or ""),
        str(question or ""),
    ])
    slot_tokens = _content_tokens(slot_text)
    if not slot_tokens:
        return False
    for claim in support_claims:
        claim_text = str(claim.get("text") or "")
        claim_tokens = _content_tokens(claim_text)
        if not claim_tokens:
            continue
        overlap = slot_tokens.intersection(claim_tokens)
        if overlap and not _looks_like_intermediate_entity_claim(claim_text, slot_text):
            return True
    return False


def _slot_repair_blocked_reason(
    slot: Dict[str, Any],
    support_claims: List[Dict[str, Any]],
    question: Optional[str],
) -> str:
    role = str(slot.get("slot_role") or "")
    if role not in {"final_answer", "direct_answer"}:
        return "non_final_slot"
    slot_text = " ".join([str(slot.get("description") or ""), str(question or "")])
    slot_tokens = _content_tokens(slot_text)
    for claim in support_claims:
        claim_text = str(claim.get("text") or "")
        if not slot_tokens.intersection(_content_tokens(claim_text)):
            continue
        if _looks_like_intermediate_entity_claim(claim_text, slot_text):
            return "intermediate_entity_claim"
    return "insufficient_keyword_overlap"


def _content_tokens(text: str) -> set:
    stopwords = {
        "the", "and", "or", "was", "were", "is", "are", "of", "in", "on", "at",
        "to", "for", "by", "with", "from", "that", "this", "what", "which",
        "who", "where", "when", "how", "did", "does", "was", "were", "person",
        "entity", "identified", "intermediate", "final", "answer", "name",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in stopwords
    }


def _looks_like_intermediate_entity_claim(claim_text: str, slot_text: str) -> bool:
    claim = str(claim_text or "").lower()
    slot = str(slot_text or "").lower()
    relation_markers = (
        "organisation", "organization", "award", "place", "district", "spouse",
        "father-in-law", "father", "mother", "county", "province", "border",
        "educated", "volunteered", "located", "headquarter", "headquarters",
    )
    if any(marker in slot for marker in relation_markers):
        if re.search(r"\b(?:is|was|were)\s+(?:the\s+)?(?:author|director|performer|spouse|father|mother|son|daughter|husband|wife)\b", claim):
            return True
        if re.search(r"\b(?:is|was|were)\s+(?:a|an|the)?\s*[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3}$", str(claim_text or "")):
            return True
    return False


def _dedupe_strs(values: List[Any]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
