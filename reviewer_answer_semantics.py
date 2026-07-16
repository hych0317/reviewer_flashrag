"""Deterministic answer semantics for ReviewerPipeline.

The helpers in this module are intentionally model- and pipeline-independent.
They turn the permissive reviewer state into relation-aware diagnostics and an
answer decision that can be replayed against historical intermediate results.
"""

import re
import time
import unicodedata
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple


VALID_OPERATIONS = {
    "direct_lookup",
    "bridge_lookup",
    "compare_min",
    "compare_max",
    "compare_equal",
    "unknown",
}

COMMIT_SAFE_COMPARISON_RELATIONS = {
    "birth_date",
    "creation_date",
    "death_date",
    "formation_date",
    "ordinal_rank",
    "release_date",
}

COMMIT_RELATION_VALUE_TYPES = {
    "birth_date": {0},
    "creation_date": {0},
    "death_date": {0},
    "formation_date": {0},
    "ordinal_rank": {1},
    "release_date": {0},
}

RELATION_ALIASES = {
    "birth_date": ("born", "birth", "date of birth"),
    "birth_place": ("born in", "birthplace", "place of birth"),
    "death_date": ("died", "death", "date of death"),
    "death_place": ("died in", "place of death"),
    "nationality": ("nationality", "citizen", "country", "from the same country"),
    "occupation": ("occupation", "profession", "musician", "musicians", "actor", "actors", "director", "directors", "writer", "writers", "singer", "singers"),
    "headquarters": ("headquartered", "headquarters", "based in"),
    "location": ("located", "location", "city", "county", "country", "situated", "place"),
    "parent": ("father", "mother", "parent"),
    "child": ("child", "son", "daughter"),
    "spouse": ("spouse", "husband", "wife", "married"),
    "director": ("director", "directed by", "directed"),
    "release_date": ("released", "came out", "premiered", "published", "opened"),
    "end_date": ("end", "ended", "end date", "concluded", "abolished", "dissolved", "ceased"),
    "capacity": ("capacity", "seated", "seats"),
    "membership": ("member", "conference", "belongs to", "part of"),
    "education": ("studied", "study", "attended", "educated", "school", "college", "university"),
    "employment": ("worked for", "employed by", "employer", "software engineer"),
    "ownership": ("owner", "owners", "owned by", "ownership"),
    "succession": ("succeeded", "successor", "replaced", "merged with"),
    "creation_date": ("created", "creation date"),
    "notable_work": ("notable work", "work did", "created", "wrote", "write for"),
    "role": ("played", "portrayed", "voiced", "voice of", "character"),
    "ordinal_rank": ("singles ranking", "ranking", "rank"),
    "formation_date": ("formed", "formation"),
    "material": ("made out of", "made of", "composed of", "constructed from", "material"),
    "award_title": ("voted", "awarded", "named", "honoured", "honored"),
    "destination": ("flow into", "flows into", "flow to", "mouth of", "drains into", "harbour"),
}

CANONICAL_RELATION_NAMES = {
    "date_of_birth": "birth_date",
    "person_birth_date": "birth_date",
    "place_of_birth": "birth_place",
    "date_of_death": "death_date",
    "person_death_date": "death_date",
    "place_of_death": "death_place",
    "film_director": "director",
    "seated_capacity": "capacity",
    "seating_capacity": "capacity",
    "abolition_date": "end_date",
}

MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
    "from", "is", "was", "were", "are", "did", "does", "do", "has", "have",
    "what", "which", "who", "where", "when", "how", "and", "or", "both",
    "same", "first", "earlier", "later", "older", "younger", "more", "less",
    "newer", "newest", "oldest", "latest", "greater", "higher", "larger",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).replace("_", " ")
    return " ".join(text.split())


def _provenance_title_match_status(title: Any, entity: Any) -> str:
    raw_title = str(title or "").strip().strip('"\'')
    raw_entity = str(entity or "").strip()
    if not raw_title or not raw_entity:
        return "unknown"
    if normalize_text(raw_title) == normalize_text(raw_entity):
        return "compatible"
    title_without_disambiguator = re.sub(r"\s*\([^)]*\)\s*$", "", raw_title).strip()
    if normalize_text(title_without_disambiguator) == normalize_text(raw_entity):
        return "compatible"

    title_tokens = set(normalize_text(raw_title).split())
    entity_tokens = set(normalize_text(raw_entity).split())
    if entity_tokens and entity_tokens < title_tokens:
        return "incompatible"
    return "unknown"


def _numeric_literals(value: Any) -> set:
    literals = set()
    for match in re.findall(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?", str(value or "")):
        try:
            literals.add(float(match.replace(",", "")))
        except ValueError:
            continue
    return literals


def _numeric_value_is_ambiguous(
    claim: Optional[Dict[str, Any]],
    raw_value: Any,
    parsed_value: Any,
    question: str,
) -> bool:
    if not claim or not isinstance(parsed_value, tuple) or parsed_value[0] != 1:
        return False
    claim_numbers = _numeric_literals(claim.get("text"))
    if len(claim_numbers) <= 1:
        return False
    relevant_numbers = claim_numbers - _numeric_literals(question)
    return bool(_numeric_literals(raw_value)) and len(relevant_numbers) > 1


def _content_tokens(value: Any) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def infer_relation_family(text: str, answer_type: Optional[str] = None) -> str:
    lower = str(text or "").lower()
    answer_type = str(answer_type or "").lower()
    normalized_relation = normalize_text(lower).replace(" ", "_")
    if normalized_relation in CANONICAL_RELATION_NAMES:
        return CANONICAL_RELATION_NAMES[normalized_relation]
    if normalized_relation in RELATION_ALIASES:
        return normalized_relation
    if re.search(r"\b(made out of|made of|composed of|constructed from)\b", lower):
        return "material"
    if re.search(r"\b(voted|awarded|named|honou?red)\b", lower):
        return "award_title"
    if re.search(r"\b(flow(?:s|ed)?\s+(?:in)?to|drains?\s+into|harbou?r)\b", lower):
        return "destination"
    if re.search(r"\bwhen\b.*\b(opened|released|premiered|published)\b", lower):
        return "release_date"
    if re.search(r"\bwhen\b.*\b(end|ended|concluded)\b", lower):
        return "end_date"
    if ("place of death" in lower or re.search(r"\bwhere\b.*\bdie", lower)):
        return "death_place"
    if ("place of birth" in lower or "birthplace" in lower or re.search(r"\bwhere\b.*\bborn", lower)):
        return "birth_place"
    if re.search(r"\b(died|death)\b", lower):
        return "death_date" if answer_type in {"date", "number"} or re.search(r"\b(first|earlier|later|when)\b", lower) else "death_place"
    if re.search(r"\b(born|birth)\b", lower):
        return "birth_date" if answer_type in {"date", "number"} or re.search(r"\b(first|earlier|later|older|younger|when)\b", lower) else "birth_place"
    for family, aliases in RELATION_ALIASES.items():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lower) for alias in aliases):
            return family
    return "generic"


def _extract_or_options(question: str) -> List[str]:
    text = str(question or "").strip().rstrip("?")
    aux_match = re.match(
        r"^(?:was|were|is|are|did|do|does)\s+(.+?)\s+or\s+(.+?)\s+"
        r"(?:born|died|released|opened|located|from|older|younger|first|earlier|later)\b",
        text,
        re.IGNORECASE,
    )
    if aux_match:
        return [part.strip(" \"'") for part in aux_match.groups()]

    # The comparison marker identifies the option-list boundary without
    # treating commas inside titles as separators.
    marker_match = re.search(
        r"\b(?:first|earlier|older|oldest|less|fewer|smaller|lowest|lower|shorter|"
        r"most\s+recent(?:ly)?|latest|later|younger|newer|newest|more|larger|greater|higher|highest|longer)\b"
        r"[^,?]*,\s*(.+)\s+or\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if marker_match:
        return [part.strip(" \"'") for part in marker_match.groups()]

    which_match = re.match(
        r"^(?:which|what|who)\b.+?,\s*(.+)\s+or\s+(.+?),\s*"
        r"(?:has|have|had|was|were|is|are|did|does|do)\b",
        text,
        re.IGNORECASE,
    )
    if which_match:
        return [part.strip(" \"'") for part in which_match.groups()]
    return []


def _extract_and_options(question: str) -> List[str]:
    text = str(question or "").strip().rstrip("?")
    match = re.match(
        r"^(?:are|were|is|was|do|did|does)\s+(.+?)\s+and\s+(.+?)\s+"
        r"(?:both|of the same|from the same|located|musicians?|actors?|directors?|writers?)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return []
    return [part.strip(" \"'") for part in match.groups()]


def extract_comparison_options(question: str, slots: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    question_options = _extract_or_options(question) or _extract_and_options(question)
    if len(question_options) >= 2:
        return question_options
    slot_options = [
        str(slot.get("comparison_option") or "").strip()
        for slot in slots or []
        if slot.get("slot_role") == "comparison_operand" and str(slot.get("comparison_option") or "").strip()
    ]
    if len(slot_options) >= 2:
        return slot_options
    return []


def _infer_option_answer_type(question: str) -> str:
    lower = str(question or "").lower().strip()
    if lower.startswith(("who ", "who,", "was ", "were ")) and not re.search(r"\b(film|movie|book|song|album)\b", lower):
        return "person"
    if re.search(r"\b(film|movie|book|novel|song|album|series|mall|company)\b", lower):
        return "title_or_entity"
    if re.search(r"\b(city|country|place|location)\b", lower):
        return "location"
    return "title_or_entity"


def build_answer_intent(
    question: str,
    question_type: Optional[str] = None,
    answer_type: Optional[str] = None,
    slots: Optional[List[Dict[str, Any]]] = None,
    parsed_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lower = str(question or "").lower()
    question_type = str(question_type or "single_hop")
    options = extract_comparison_options(question, slots)
    deterministic_operation = "unknown"
    answer_role = "object"

    if question_type == "comparison" or options:
        if re.search(r"\b(same|both)\b", lower):
            deterministic_operation = "compare_equal"
            answer_role = "boolean"
        elif re.search(r"\bbest\b.*\b(?:ranking|rank)\b", lower):
            deterministic_operation = "compare_min"
            answer_role = "option"
        elif re.search(r"\b(first|earlier|older|oldest|less|fewer|smaller|lowest|lower|shorter)\b", lower):
            deterministic_operation = "compare_min"
            answer_role = "option"
        elif re.search(r"\b(most recent(?:ly)?|latest|later|younger|newer|newest|more|larger|greater|higher|highest|longer)\b", lower):
            deterministic_operation = "compare_max"
            answer_role = "option"
    elif question_type == "bridge":
        deterministic_operation = "bridge_lookup"
    elif question_type == "single_hop":
        deterministic_operation = "direct_lookup"

    parsed_intent = parsed_intent or {}
    parsed_operation = str(parsed_intent.get("operation") or "")
    parsed_operation_valid = parsed_operation in VALID_OPERATIONS and parsed_operation != "unknown"
    intent_conflict = bool(
        parsed_operation_valid
        and deterministic_operation != "unknown"
        and parsed_operation != deterministic_operation
    )
    if deterministic_operation != "unknown":
        operation = deterministic_operation
        intent_source = "deterministic_question_pattern"
    elif parsed_operation_valid:
        operation = parsed_operation
        intent_source = "analyzer"
    else:
        operation = "unknown"
        intent_source = "unknown"
    if not options:
        options = [str(item).strip() for item in parsed_intent.get("option_entities") or [] if str(item).strip()]

    expected_answer_type = str(answer_type or "phrase")
    if operation == "compare_equal":
        expected_answer_type = "yes_no"
    elif operation in {"compare_min", "compare_max"}:
        expected_answer_type = _infer_option_answer_type(question)

    semantic_roles = {"comparison_operand"} if operation.startswith("compare_") else {"final_answer", "direct_answer"}
    slot_relations = [
        str(slot.get("target_relation") or "").strip()
        for slot in slots or []
        if slot.get("slot_role") in semantic_roles
        and str(slot.get("target_relation") or "").strip().lower() not in {"", "generic", "unknown", "answer", "final_answer"}
    ]
    analyzer_relation = str(parsed_intent.get("target_relation") or "").strip()
    analyzer_family = infer_relation_family(analyzer_relation, expected_answer_type) if analyzer_relation else "generic"
    focused_family = infer_relation_family(_focused_relation_clause(question), expected_answer_type)
    slot_family = infer_relation_family(slot_relations[-1], expected_answer_type) if slot_relations else "generic"
    ordered_date_families = {
        "birth_date", "creation_date", "death_date", "end_date",
        "formation_date", "release_date",
    }
    focused_override = bool(
        operation in {"compare_min", "compare_max"}
        and focused_family in ordered_date_families
        and slot_family != focused_family
    )
    if focused_override:
        target_relation = focused_family
        relation_source = "focused_question_override"
    elif slot_relations:
        target_relation = slot_relations[-1]
        relation_source = "semantic_slot"
    elif analyzer_relation and analyzer_family != "generic":
        target_relation = analyzer_relation
        relation_source = "analyzer"
    elif focused_family != "generic":
        target_relation = focused_family
        relation_source = "focused_question_clause"
    else:
        target_relation = analyzer_relation or "generic"
        relation_source = "generic"
    relation_family = infer_relation_family(target_relation, expected_answer_type)

    return {
        "operation": operation,
        "target_relation": target_relation,
        "relation_family": relation_family,
        "answer_role": answer_role,
        "expected_answer_type": expected_answer_type,
        "option_entities": options[:4],
        "intent_source": intent_source,
        "intent_conflict": intent_conflict,
        "parsed_operation": parsed_operation or None,
        "deterministic_operation": deterministic_operation,
        "relation_source": relation_source,
        "relation_override": focused_override,
    }


def build_comparison_output_contract(
    question: str,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the derived comparison output without creating an evidence slot."""
    state = state or {}
    if str(state.get("question_type") or "") != "comparison":
        return {}
    intent = build_answer_intent(
        question,
        state.get("question_type"),
        state.get("answer_type"),
        state.get("answer_slots") or [],
        state.get("answer_intent"),
    )
    operation = str(intent.get("operation") or "unknown")
    output_role = "boolean" if operation == "compare_equal" else "option"
    options = list(intent.get("option_entities") or [])[:4]
    normalized_question = normalize_text(question)
    if len(options) != 2 or any(
        not normalize_text(option) or normalize_text(option) not in normalized_question
        for option in options
    ):
        return {}
    return {
        "operation": operation,
        "option_entities": options,
        "output_role": output_role,
        "expected_answer_type": intent.get("expected_answer_type") or state.get("answer_type"),
        "derived_output": True,
    }


def validate_final_comparison_commit(
    relation_trace: Optional[Dict[str, Any]],
    output_contract: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate a comparison candidate for finalization-only deterministic commit."""
    relation_trace = relation_trace or {}
    output_contract = output_contract or {}
    result = str(relation_trace.get("counterfactual_answer_candidate") or "").strip()
    branches = relation_trace.get("comparison_branches") or []
    answer_intent = relation_trace.get("answer_intent") or {}
    relation_family = str(answer_intent.get("relation_family") or "generic")

    def rejected(reason: str) -> Dict[str, Any]:
        return {
            "eligible": False,
            "candidate": result or None,
            "committed_claim_ids": [],
            "blocked_reason": reason,
            "commit_confidence": "blocked",
            "relation_family": relation_family,
            "relation_value_type_valid": (
                False if reason == "relation_value_type_mismatch" else None
            ),
        }

    if not output_contract:
        return rejected("missing_output_contract")
    if not relation_trace.get("comparison_branch_executable"):
        return rejected(
            str(relation_trace.get("comparison_branch_failure_reason") or "branch_not_executable")
        )
    if len(branches) != 2:
        return rejected("requires_exactly_two_branches")
    if not result:
        return rejected("missing_comparison_result")

    options = [str(option).strip() for option in output_contract.get("option_entities") or []]
    if output_contract.get("output_role") == "boolean":
        if normalize_text(result) not in {"yes", "no"}:
            return rejected("result_is_not_boolean")
        canonical_result = normalize_text(result)
    else:
        matching_options = [option for option in options if normalize_text(option) == normalize_text(result)]
        if len(matching_options) != 1:
            return rejected("result_is_not_unique_question_option")
        canonical_result = matching_options[0]

    committed_claim_ids: List[str] = []
    for branch in branches:
        if branch.get("status") != "supported":
            return rejected("incomplete_or_conflicted_branch")
        path_type = str(branch.get("path_type") or "")
        if path_type not in {"direct", "one_hop"}:
            return rejected("slot_associated_path_rejected")
        if not branch.get("value_claim_id") or not branch.get("provenance_doc_ids"):
            return rejected("incomplete_branch_provenance")
        if path_type == "one_hop" and not branch.get("intermediate_claim_id"):
            return rejected("incomplete_one_hop_provenance")
        if (
            branch.get("provenance_entity_match_status") == "incompatible"
            or branch.get("provenance_entity_compatible") is False
        ):
            return rejected("provenance_entity_mismatch")
        if branch.get("numeric_value_ambiguous"):
            return rejected("ambiguous_numeric_provenance")
        committed_claim_ids.extend(str(claim_id) for claim_id in branch.get("claim_ids") or [])

    if relation_family not in COMMIT_SAFE_COMPARISON_RELATIONS:
        return rejected("relation_family_not_commit_safe")
    expected_value_types = COMMIT_RELATION_VALUE_TYPES.get(relation_family) or set()
    if expected_value_types and any(
        branch.get("value_type") not in expected_value_types for branch in branches
    ):
        return rejected("relation_value_type_mismatch")

    return {
        "eligible": True,
        "candidate": canonical_result,
        "committed_claim_ids": list(dict.fromkeys(committed_claim_ids)),
        "blocked_reason": None,
        "commit_confidence": "strict",
        "relation_family": relation_family,
        "relation_value_type_valid": True,
    }


def _focused_relation_clause(question: str) -> str:
    text = str(question or "").strip().rstrip("?")
    for option in _extract_or_options(text) or _extract_and_options(text):
        text = re.sub(re.escape(option), " ", text, count=1, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,")


def repair_answer_slot_topology(
    slots: List[Dict[str, Any]],
    question: str,
    question_type: Optional[str],
) -> List[Dict[str, Any]]:
    """Repair only slot roles/dependencies; evidence status is never promoted."""
    repaired = deepcopy(slots or [])
    used_ids = {str(slot.get("slot_id")) for slot in repaired}

    def add_slot(role: str, description: str) -> Dict[str, Any]:
        index = len(repaired) + 1
        slot_id = f"slot_{index}"
        while slot_id in used_ids:
            index += 1
            slot_id = f"slot_{index}"
        used_ids.add(slot_id)
        slot = {
            "slot_id": slot_id,
            "description": description,
            "slot_role": role,
            "depends_on": [],
            "status": "missing",
            "supporting_claim_ids": [],
            "conflicting_claim_ids": [],
        }
        repaired.append(slot)
        return slot

    options = extract_comparison_options(question, repaired)
    is_comparison = str(question_type) == "comparison"
    if is_comparison:
        operands = [slot for slot in repaired if slot.get("slot_role") == "comparison_operand"]
        while len(operands) < 2:
            operands.append(add_slot("comparison_operand", f"Resolve comparison operand {len(operands) + 1}"))
        final_slots = [slot for slot in repaired if slot.get("slot_role") == "final_answer"]
        for final_slot in final_slots:
            final_slot["derived_output"] = True
            final_slot["depends_on"] = list(dict.fromkeys(
                list(final_slot.get("depends_on") or []) + [str(slot.get("slot_id")) for slot in operands[:2]]
            ))
        return repaired

    intermediates = [slot for slot in repaired if slot.get("slot_role") == "intermediate_entity"]
    existing_finals = [slot for slot in repaired if slot.get("slot_role") in {"final_answer", "direct_answer"}]
    if str(question_type) == "bridge" or (intermediates and existing_finals):
        if not intermediates:
            intermediates.append(add_slot("intermediate_entity", "Resolve the intermediate entity"))
        final_slots = [slot for slot in repaired if slot.get("slot_role") in {"final_answer", "direct_answer"}]
        final_slot = final_slots[-1] if final_slots else add_slot("final_answer", "Resolve the final answer")
        final_slot["slot_role"] = "final_answer"
        final_slot["depends_on"] = list(dict.fromkeys(
            list(final_slot.get("depends_on") or []) + [str(intermediates[-1].get("slot_id"))]
        ))
        return repaired

    if not any(slot.get("slot_role") == "direct_answer" for slot in repaired):
        final_slots = [slot for slot in repaired if slot.get("slot_role") == "final_answer"]
        if final_slots:
            final_slots[-1]["slot_role"] = "direct_answer"
        else:
            add_slot("direct_answer", "Resolve the direct answer")
    return repaired


def infer_effective_question_type(
    question_type: Optional[str],
    slots: List[Dict[str, Any]],
    question: str,
) -> Tuple[str, Optional[str]]:
    roles = [str(slot.get("slot_role") or "") for slot in slots or []]
    operand_count = sum(role == "comparison_operand" for role in roles)
    has_intermediate = "intermediate_entity" in roles
    has_final = any(role in {"final_answer", "direct_answer"} for role in roles)
    if operand_count >= 2 or len(extract_comparison_options(question, slots)) == 2:
        effective = "comparison"
        reason = "comparison_slot_topology"
    elif has_intermediate and has_final:
        effective = "bridge"
        reason = "intermediate_final_slot_topology"
    else:
        effective = str(question_type or "single_hop")
        reason = None
    return effective, reason if effective != str(question_type or "single_hop") else None


def annotate_slots_with_intent(slots: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotated = deepcopy(slots or [])
    options = list(intent.get("option_entities") or [])
    operand_index = 0
    for slot in annotated:
        role = str(slot.get("slot_role") or "")
        slot.setdefault("target_relation", intent.get("target_relation"))
        slot.setdefault("relation_family", intent.get("relation_family"))
        slot.setdefault("expected_answer_type", intent.get("expected_answer_type"))
        if role == "comparison_operand" and operand_index < len(options):
            slot["comparison_option"] = options[operand_index]
            operand_index += 1
    return annotated


def _doc_title_map(docs: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    result = {}
    for index, doc in enumerate(docs or []):
        doc_id = str(doc.get("id", f"doc_{index}"))
        title = str(doc.get("title") or "")
        if not title:
            content = str(doc.get("contents", doc.get("content", ""))).strip()
            title = content.splitlines()[0][:120] if content else ""
        result[doc_id] = title
    return result


def _infer_subject(text: str) -> Optional[str]:
    match = re.match(
        r"^(?:The\s+)?([A-Z][A-Za-z0-9$'.-]+(?:\s+(?:of|the|and|[A-Z][A-Za-z0-9$'.-]+)){0,7})\s+"
        r"(?:is|was|were|has|had|died|was born|worked|served|directed|released|opened)\b",
        str(text or "").strip(),
    )
    return match.group(1).strip(" ,.") if match else None


def _infer_object(text: str) -> Optional[str]:
    value = str(text or "").strip()
    patterns = (
        r"\b(?:is|was|were|became|remained)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[.;]|$)",
        r"\b(?:born|died|located|based|headquartered|situated)\s+(?:in|at|near)\s+(.+?)(?:[.;]|$)",
        r"\b(?:released|opened|ended|published|premiered)\s+(?:in|on)\s+(.+?)(?:[.;]|$)",
        r"\b(?:directed|created|written|performed)\s+by\s+(.+?)(?:[.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" \"'`,.")
            if 0 < len(candidate.split()) <= 12:
                return candidate
    return None


def _entity_mentioned(entity: Any, text: Any) -> bool:
    normalized_entity = normalize_text(entity)
    normalized_text = normalize_text(text)
    if not normalized_entity or not normalized_text:
        return False
    if normalized_entity in normalized_text:
        return True
    entity_tokens = set(normalized_entity.split())
    return len(entity_tokens) >= 2 and entity_tokens.issubset(set(normalized_text.split()))


def _question_entities(question: str) -> List[str]:
    entities = []
    for match in re.finditer(
        r"(?:[A-ZÀ-ÖØ-Þ][\w'’.-]+(?:\s+(?:of|the|and|de|la|le|[A-ZÀ-ÖØ-Þ][\w'’.-]+)){0,7})",
        str(question or ""),
    ):
        entity = match.group(0).strip(" ,.?\"'")
        if entity:
            entities.append(entity)
    return entities


def _relation_matches(intent: Dict[str, Any], claim: Dict[str, Any]) -> bool:
    family = str(intent.get("relation_family") or "generic")
    predicate_text = " ".join((str(claim.get("predicate") or ""), str(claim.get("text") or "")))
    if family != "generic":
        relation_answer_type = intent.get("expected_answer_type")
        if family in {
            "birth_date", "creation_date", "death_date", "end_date",
            "formation_date", "release_date",
        }:
            relation_answer_type = "date"
        inferred_family = infer_relation_family(predicate_text, relation_answer_type)
        if family == "creation_date" and inferred_family == "release_date":
            return True
        return inferred_family == family
    target_tokens = _content_tokens(intent.get("target_relation"))
    return bool(target_tokens and target_tokens.intersection(_content_tokens(predicate_text)))


def _default_answer_type_match(candidate: str, answer_type: str) -> bool:
    text = str(candidate or "").strip()
    answer_type = str(answer_type or "").lower()
    if not text:
        return False
    if answer_type == "yes_no":
        return text.lower() in {"yes", "no"}
    if answer_type == "number":
        return bool(re.search(r"\d", text))
    if answer_type == "date":
        return bool(re.search(r"\b\d{4}\b|\b(" + "|".join(MONTHS) + r")\b", text, re.IGNORECASE))
    if answer_type in {"person", "location", "title_or_entity"}:
        return len(text.split()) <= 10 and bool(re.search(r"[A-Za-z]", text))
    return len(text.split()) <= 16


def _extract_resolved_entities(
    state: Dict[str, Any],
    candidate_extractor: Optional[Callable[[str, Optional[str], Optional[str]], Optional[str]]],
    question: str,
) -> List[Dict[str, Any]]:
    slots = state.get("answer_slots") or []
    intermediate_ids = {
        str(slot.get("slot_id"))
        for slot in slots
        if slot.get("slot_role") == "intermediate_entity"
    }
    resolved = []
    seen = set()
    for claim in state.get("claims") or []:
        if str(claim.get("slot_id")) not in intermediate_ids or claim.get("support_status") != "support":
            continue
        subject = str(claim.get("subject") or "").strip()
        object_value = str(claim.get("object") or "").strip()
        subject_in_question = bool(subject and _entity_mentioned(subject, question))
        object_in_question = bool(object_value and _entity_mentioned(object_value, question))
        if object_in_question and not subject_in_question:
            entity = subject
            direction = "object_anchor_to_subject"
        elif subject_in_question and not object_in_question:
            entity = object_value
            direction = "subject_anchor_to_object"
        elif object_value and not subject:
            entity = object_value
            direction = "object_only"
        elif subject and not object_value:
            entity = subject
            direction = "subject_only"
        else:
            entity = ""
            direction = "ambiguous_endpoints"
        if not entity and not subject and not object_value:
            entity = _infer_object(claim.get("text", "")) or ""
        if not entity and not subject and not object_value and candidate_extractor:
            entity = candidate_extractor(claim.get("text", ""), "title_or_entity", question) or ""
        key = normalize_text(entity)
        if not key or key in seen:
            continue
        seen.add(key)
        resolved.append({
            "entity": entity,
            "normalized": key,
            "slot_id": claim.get("slot_id"),
            "claim_id": claim.get("claim_id"),
            "binding_direction": direction,
        })
    return resolved


def resolve_claim_binding(
    claim: Dict[str, Any],
    anchor_entities: List[str],
    question: str,
) -> Dict[str, Any]:
    subject = str(claim.get("subject") or "").strip() or _infer_subject(claim.get("text", "")) or ""
    object_value = str(claim.get("object") or "").strip() or _infer_object(claim.get("text", "")) or ""
    anchors = [str(anchor).strip() for anchor in anchor_entities or [] if str(anchor).strip()]
    anchor_haystacks = anchors or [question]
    subject_matches = [anchor for anchor in anchor_haystacks if subject and _entity_mentioned(subject, anchor)]
    object_matches = [anchor for anchor in anchor_haystacks if object_value and _entity_mentioned(object_value, anchor)]

    if subject_matches and not object_matches:
        return {
            "candidate": object_value or None,
            "binding_direction": "subject_anchor_to_object",
            "anchor_entity": subject_matches[0],
            "anchor_endpoint": "subject",
            "candidate_endpoint": "object",
            "entity_binding_valid": bool(object_value),
            "entity_binding_failure_reason": None if object_value else "missing_object_endpoint",
            "subject": subject or None,
            "object": object_value or None,
        }
    if object_matches and not subject_matches:
        return {
            "candidate": subject or None,
            "binding_direction": "object_anchor_to_subject",
            "anchor_entity": object_matches[0],
            "anchor_endpoint": "object",
            "candidate_endpoint": "subject",
            "entity_binding_valid": bool(subject),
            "entity_binding_failure_reason": None if subject else "missing_subject_endpoint",
            "subject": subject or None,
            "object": object_value or None,
        }
    failure_reason = "both_endpoints_match_anchor" if subject_matches and object_matches else "no_endpoint_matches_anchor"
    return {
        "candidate": None,
        "binding_direction": "ambiguous",
        "anchor_entity": (subject_matches or object_matches or [None])[0],
        "anchor_endpoint": None,
        "candidate_endpoint": None,
        "entity_binding_valid": False,
        "entity_binding_failure_reason": failure_reason,
        "subject": subject or None,
        "object": object_value or None,
    }


def _parse_orderable(value: str) -> Optional[Tuple[int, int, int, int, int]]:
    text = str(value or "")
    month_pattern = "|".join(name.title() for name in MONTHS)
    iso_date = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_date:
        return (0, int(iso_date.group(1)), int(iso_date.group(2)), int(iso_date.group(3)), 3)
    full_date = re.search(
        rf"\b({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{3,4}})\b",
        text,
        re.IGNORECASE,
    )
    if full_date:
        month = MONTHS[full_date.group(1).lower()]
        return (0, int(full_date.group(3)), month, int(full_date.group(2)), 3)
    day_month_year = re.search(
        rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{3,4}})\b",
        text,
        re.IGNORECASE,
    )
    if day_month_year:
        return (
            0,
            int(day_month_year.group(3)),
            MONTHS[day_month_year.group(2).lower()],
            int(day_month_year.group(1)),
            3,
        )
    month_year = re.search(rf"\b({month_pattern})\s+(\d{{3,4}})\b", text, re.IGNORECASE)
    if month_year:
        return (0, int(month_year.group(2)), MONTHS[month_year.group(1).lower()], 0, 2)
    years = {int(item) for item in re.findall(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)", text)}
    if len(years) == 1:
        return (0, years.pop(), 0, 0, 1)
    numbers = {
        float(item.replace(",", ""))
        for item in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", text)
    }
    if len(numbers) == 1:
        number = numbers.pop()
        return (1, int(number * 1000), 0, 0, 1)
    return None


def _comparison_value(claim: Dict[str, Any], operation: str) -> Any:
    value = str(claim.get("object") or "").strip() or _infer_object(claim.get("text", "")) or str(claim.get("text") or "")
    if operation in {"compare_min", "compare_max"}:
        return _parse_orderable(value)
    return normalize_text(value)


def _execute_comparison_strict_slots(
    intent: Dict[str, Any],
    state: Dict[str, Any],
    docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    operation = str(intent.get("operation") or "unknown")
    if operation not in {"compare_min", "compare_max", "compare_equal"}:
        return {
            "status": "not_applicable", "operator": operation, "operands": [],
            "comparison_lineage": [], "operand_binding_status": "not_applicable",
            "strict_comparison_executable": False, "result": None,
            "failure_reason": None, "rejection_reasons": [], "committed_claim_ids": [],
        }

    options = [str(option).strip() for option in intent.get("option_entities") or [] if str(option).strip()]
    slots = state.get("answer_slots") or []
    slots_by_id = {str(slot.get("slot_id")): slot for slot in slots}
    title_by_doc_id = _doc_title_map(docs)
    available_doc_ids = set(title_by_doc_id) if docs else set()
    content_by_doc_id = {
        str(doc.get("id", f"doc_{index}")): str(doc.get("contents", doc.get("content", "")))
        for index, doc in enumerate(docs or [])
    }
    claims = [claim for claim in state.get("claims") or [] if claim.get("support_status") == "support"]

    def source_available(claim: Dict[str, Any]) -> bool:
        return not available_doc_ids or str(claim.get("doc_id") or "") in available_doc_ids

    def doc_supports_claim(doc_id: str, claim: Dict[str, Any]) -> bool:
        title = title_by_doc_id.get(doc_id, "")
        content = content_by_doc_id.get(doc_id, "")
        subject = claim.get("subject")
        value = claim.get("object")
        return bool(
            doc_id in available_doc_ids
            and (_entity_mentioned(subject, title) or _entity_mentioned(subject, content))
            and value not in (None, "")
            and normalize_text(value) in normalize_text(content)
            and _relation_matches(intent, {
                "predicate": "",
                "text": content,
            })
        )

    def provenance_doc_id(claim: Dict[str, Any]) -> Optional[str]:
        if not docs:
            return str(claim.get("doc_id") or "") or None
        preferred = str(claim.get("doc_id") or "")
        return preferred if doc_supports_claim(preferred, claim) else None

    def alternative_provenance_doc_ids(claim: Optional[Dict[str, Any]]) -> List[str]:
        if not claim or not docs:
            return []
        preferred = str(claim.get("doc_id") or "")
        return [
            doc_id for doc_id in sorted(available_doc_ids)
            if doc_id != preferred and doc_supports_claim(doc_id, claim)
        ][:8]

    def structured_value_claim(claim: Dict[str, Any]) -> bool:
        return bool(
            claim.get("claim_id")
            and claim.get("subject")
            and claim.get("predicate")
            and claim.get("object") not in (None, "")
            and source_available(claim)
            and _relation_matches(intent, claim)
        )

    def depends_on(slot_id: Any, ancestor_id: Any) -> bool:
        pending = list((slots_by_id.get(str(slot_id)) or {}).get("depends_on") or [])
        visited = set()
        while pending:
            current = str(pending.pop())
            if current == str(ancestor_id):
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend((slots_by_id.get(current) or {}).get("depends_on") or [])
        return False

    lineage = []
    for option in options[:2]:
        rejection_reasons = []
        direct = [
            claim for claim in claims
            if structured_value_claim(claim)
            and normalize_text(claim.get("subject")) == normalize_text(option)
            and (slots_by_id.get(str(claim.get("slot_id"))) or {}).get("slot_role") == "comparison_operand"
            and normalize_text((slots_by_id.get(str(claim.get("slot_id"))) or {}).get("comparison_option")) == normalize_text(option)
        ]
        intermediate_claim = None
        value_claims = direct
        intermediate_entity = option
        if not value_claims:
            bridges = [
                claim for claim in claims
                if claim.get("claim_id") and claim.get("subject") and claim.get("object") not in (None, "")
                and source_available(claim)
                and normalize_text(claim.get("subject")) == normalize_text(option)
            ]
            nested = []
            for bridge_claim in bridges:
                entity = str(bridge_claim.get("object") or "").strip()
                for claim in claims:
                    if not structured_value_claim(claim):
                        continue
                    if normalize_text(claim.get("subject")) != normalize_text(entity):
                        continue
                    value_slot = slots_by_id.get(str(claim.get("slot_id"))) or {}
                    if value_slot.get("slot_role") != "comparison_operand":
                        continue
                    if normalize_text(value_slot.get("comparison_option")) != normalize_text(option):
                        continue
                    if not depends_on(claim.get("slot_id"), bridge_claim.get("slot_id")):
                        continue
                    nested.append((bridge_claim, claim))
            if nested:
                intermediate_claim = nested[0][0]
                intermediate_entity = str(intermediate_claim.get("object") or "").strip()
                value_claims = [value_claim for _, value_claim in nested]

        parsed = [(_comparison_value(claim, operation), claim) for claim in value_claims]
        parsed = [(value, claim) for value, claim in parsed if value not in (None, "")]
        unique_values = {str(value) for value, _ in parsed}
        if not value_claims:
            rejection_reasons.append("missing_option_value_lineage")
        elif len(unique_values) != 1:
            rejection_reasons.append("ambiguous_option_value")
        value = parsed[0][0] if len(unique_values) == 1 else None
        value_claim = parsed[0][1] if len(unique_values) == 1 else None
        if docs and value_claim and not provenance_doc_id(value_claim):
            rejection_reasons.append("original_value_claim_source_not_supportive")
        if docs and intermediate_claim and not provenance_doc_id(intermediate_claim):
            rejection_reasons.append("original_intermediate_claim_source_not_supportive")
        value_slot = slots_by_id.get(str(value_claim.get("slot_id"))) if value_claim else {}
        if value_slot and (value_slot.get("status") == "conflicted" or value_slot.get("conflicting_claim_ids")):
            rejection_reasons.append("value_slot_conflicted")
        lineage.append({
            "option": option,
            "option_slot_id": intermediate_claim.get("slot_id") if intermediate_claim else (value_claim.get("slot_id") if value_claim else None),
            "intermediate_entity": intermediate_entity or None,
            "intermediate_claim_id": intermediate_claim.get("claim_id") if intermediate_claim else None,
            "value_claim_id": value_claim.get("claim_id") if value_claim else None,
            "intermediate_provenance_doc_id": provenance_doc_id(intermediate_claim) if intermediate_claim else None,
            "value_provenance_doc_id": provenance_doc_id(value_claim) if value_claim else None,
            "alternative_intermediate_provenance_doc_ids": alternative_provenance_doc_ids(intermediate_claim),
            "alternative_value_provenance_doc_ids": alternative_provenance_doc_ids(value_claim),
            "provenance_valid": bool(value_claim and provenance_doc_id(value_claim)),
            "provenance_failure_reason": (
                None if value_claim and provenance_doc_id(value_claim)
                else "original_claim_source_not_supportive" if value_claim else "missing_value_claim"
            ),
            "value": value,
            "binding_valid": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "claim_ids": [
                str(claim_id) for claim_id in (
                    intermediate_claim.get("claim_id") if intermediate_claim else None,
                    value_claim.get("claim_id") if value_claim else None,
                ) if claim_id
            ],
        })

    global_rejections = []
    if len(options) != 2:
        global_rejections.append("requires_exactly_two_options")
    if intent.get("intent_conflict"):
        global_rejections.append("intent_conflict")
    if str(intent.get("relation_family") or "generic") == "generic":
        global_rejections.append("generic_target_relation")
    valid_lineage = [item for item in lineage if item.get("binding_valid")]
    if len(valid_lineage) == 2:
        subjects = [normalize_text(item.get("intermediate_entity")) for item in valid_lineage]
        value_claim_ids = [str(item.get("value_claim_id")) for item in valid_lineage]
        if len(set(subjects)) != 2:
            global_rejections.append("operand_subject_reused")
        if len(set(value_claim_ids)) != 2:
            global_rejections.append("value_claim_reused")
        value_types = {
            item.get("value")[0] if isinstance(item.get("value"), tuple) else type(item.get("value")).__name__
            for item in valid_lineage
        }
        if len(value_types) != 1:
            global_rejections.append("operand_type_mismatch")
        date_precisions = {
            item.get("value")[4]
            for item in valid_lineage
            if isinstance(item.get("value"), tuple) and item.get("value")[0] == 0
        }
        if len(date_precisions) > 1:
            global_rejections.append("operand_date_precision_mismatch")

    strict_executable = len(lineage) == 2 and all(item.get("binding_valid") for item in lineage) and not global_rejections
    operands = [
        {
            "slot_id": item.get("option_slot_id"),
            "option": item.get("option"),
            "value": item.get("value"),
            "claim_ids": item.get("claim_ids") or [],
        }
        for item in lineage
    ]
    if not strict_executable:
        failure_reason = global_rejections[0] if global_rejections else "missing_or_ambiguous_operand"
        return {
            "status": "incomplete_operands",
            "operator": operation,
            "operands": operands,
            "comparison_lineage": lineage,
            "operand_binding_status": "invalid",
            "strict_comparison_executable": False,
            "result": None,
            "failure_reason": failure_reason,
            "rejection_reasons": global_rejections,
            "committed_claim_ids": [],
        }

    values = [item["value"] for item in operands]
    if operation == "compare_equal":
        result = "yes" if all(value == values[0] for value in values[1:]) else "no"
    else:
        target = min(values) if operation == "compare_min" else max(values)
        winners = [item for item in operands if item["value"] == target]
        if len(winners) != 1:
            return {
                "status": "ambiguous_result", "operator": operation, "operands": operands,
                "comparison_lineage": lineage, "operand_binding_status": "ambiguous",
                "strict_comparison_executable": False, "result": None,
                "failure_reason": "comparison_tie", "rejection_reasons": ["comparison_tie"],
                "committed_claim_ids": [],
            }
        result = winners[0]["option"]
    committed_claim_ids = [claim_id for item in lineage for claim_id in item.get("claim_ids") or []]
    return {
        "status": "executed",
        "operator": operation,
        "operands": operands,
        "comparison_lineage": lineage,
        "operand_binding_status": "valid",
        "strict_comparison_executable": True,
        "result": result,
        "failure_reason": None,
        "rejection_reasons": [],
        "committed_claim_ids": list(dict.fromkeys(str(item) for item in committed_claim_ids)),
    }


def _comparison_entities_match(left: Any, right: Any) -> bool:
    return bool(
        normalize_text(left) == normalize_text(right)
        or _entity_mentioned(left, right)
        or _entity_mentioned(right, left)
    )


def build_comparison_branches(
    question: str,
    state: Dict[str, Any],
    docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build at most one-hop option-to-value branches from supported claims."""
    started = time.perf_counter()
    slots = state.get("answer_slots") or []
    intent = build_answer_intent(
        question,
        state.get("question_type"),
        state.get("answer_type"),
        slots,
        state.get("answer_intent"),
    )
    operation = str(intent.get("operation") or "unknown")
    options = [
        str(option).strip()
        for option in intent.get("option_entities") or []
        if str(option).strip()
    ][:2]
    slots_by_id = {str(slot.get("slot_id")): slot for slot in slots}
    claims = [
        claim for claim in state.get("claims") or []
        if claim.get("support_status") == "support"
        and claim.get("claim_id")
        and claim.get("subject")
        and claim.get("object") not in (None, "")
    ]
    content_by_doc_id = {
        str(doc.get("id", f"doc_{index}")): str(doc.get("contents", doc.get("content", "")))
        for index, doc in enumerate(docs or [])
    }
    title_by_doc_id = _doc_title_map(docs)

    def claim_title_match_status(claim: Dict[str, Any], entity: Any) -> str:
        if not docs:
            return "unknown"
        title = title_by_doc_id.get(str(claim.get("doc_id") or ""), "")
        return _provenance_title_match_status(title, entity)

    def claim_title_matches_entity(claim: Dict[str, Any], entity: Any) -> Optional[bool]:
        status = claim_title_match_status(claim, entity)
        if status == "compatible":
            return True
        if status == "incompatible":
            return False
        return None

    def combine_title_match_statuses(*statuses: str) -> str:
        if "incompatible" in statuses:
            return "incompatible"
        if statuses and all(status == "compatible" for status in statuses):
            return "compatible"
        return "unknown"

    def title_match_compatibility(status: str) -> Optional[bool]:
        if status == "compatible":
            return True
        if status == "incompatible":
            return False
        return None

    def claim_provenance_valid(claim: Dict[str, Any]) -> bool:
        doc_id = str(claim.get("doc_id") or "")
        if not doc_id:
            return False
        if not docs or doc_id not in content_by_doc_id:
            return True
        content = content_by_doc_id.get(doc_id, "")
        return bool(
            content
            and _entity_mentioned(claim.get("subject"), content)
            and _entity_mentioned(claim.get("object"), content)
        )

    def slot_conflicted(claim: Dict[str, Any]) -> bool:
        slot = slots_by_id.get(str(claim.get("slot_id"))) or {}
        return bool(slot.get("status") == "conflicted" or slot.get("conflicting_claim_ids"))

    def opposite_endpoint(claim: Dict[str, Any], anchor: Any) -> Optional[str]:
        subject = str(claim.get("subject") or "").strip()
        object_value = str(claim.get("object") or "").strip()
        subject_match = _comparison_entities_match(subject, anchor)
        object_match = _comparison_entities_match(object_value, anchor)
        if subject_match == object_match:
            return None
        return object_value if subject_match else subject

    branches = []
    for option in options:
        paths = []
        for value_claim in claims:
            if not _relation_matches(intent, value_claim) or slot_conflicted(value_claim):
                continue
            direct_value = opposite_endpoint(value_claim, option)
            if direct_value and claim_provenance_valid(value_claim):
                paths.append({
                    "path_type": "direct",
                    "intermediate_entity": None,
                    "intermediate_claim": None,
                    "value_claim": value_claim,
                    "raw_value": direct_value,
                    "provenance_entity_compatible": claim_title_matches_entity(
                        value_claim, option
                    ),
                    "provenance_entity_match_status": claim_title_match_status(
                        value_claim, option
                    ),
                })
            value_slot = slots_by_id.get(str(value_claim.get("slot_id"))) or {}
            slot_option = value_slot.get("comparison_option")
            if (
                not direct_value
                and slot_option
                and _comparison_entities_match(slot_option, option)
                and not any(
                    _comparison_entities_match(endpoint, other_option)
                    for other_option in options if not _comparison_entities_match(other_option, option)
                    for endpoint in (value_claim.get("subject"), value_claim.get("object"))
                )
                and claim_provenance_valid(value_claim)
            ):
                paths.append({
                    "path_type": "slot_associated",
                    "intermediate_entity": str(value_claim.get("subject") or "").strip() or None,
                    "intermediate_claim": None,
                    "value_claim": value_claim,
                    "raw_value": str(value_claim.get("object") or "").strip(),
                    "provenance_entity_compatible": claim_title_matches_entity(
                        value_claim, option
                    ),
                    "provenance_entity_match_status": claim_title_match_status(
                        value_claim, option
                    ),
                })

        for bridge_claim in claims:
            if _relation_matches(intent, bridge_claim) or slot_conflicted(bridge_claim):
                continue
            intermediate = opposite_endpoint(bridge_claim, option)
            if not intermediate or not claim_provenance_valid(bridge_claim):
                continue
            for value_claim in claims:
                if value_claim is bridge_claim:
                    continue
                if not _relation_matches(intent, value_claim) or slot_conflicted(value_claim):
                    continue
                raw_value = opposite_endpoint(value_claim, intermediate)
                if not raw_value or not claim_provenance_valid(value_claim):
                    continue
                if _comparison_entities_match(raw_value, option) or _comparison_entities_match(raw_value, intermediate):
                    continue
                paths.append({
                    "path_type": "one_hop",
                    "intermediate_entity": intermediate,
                    "intermediate_claim": bridge_claim,
                    "value_claim": value_claim,
                    "raw_value": raw_value,
                    "provenance_entity_compatible": title_match_compatibility(
                        combine_title_match_statuses(
                            claim_title_match_status(bridge_claim, option),
                            claim_title_match_status(value_claim, intermediate),
                        )
                    ),
                    "provenance_entity_match_status": combine_title_match_statuses(
                        claim_title_match_status(bridge_claim, option),
                        claim_title_match_status(value_claim, intermediate),
                    ),
                })

        parsed_paths = []
        for path in paths:
            synthetic_claim = dict(path["value_claim"])
            synthetic_claim["object"] = path["raw_value"]
            value = _comparison_value(synthetic_claim, operation)
            if value not in (None, ""):
                parsed_paths.append((value, path))
        values = {}
        for value, path in parsed_paths:
            values.setdefault(repr(value), []).append((value, path))

        rejection_reasons = []
        selected_value = None
        selected_path = None
        if not values:
            status = "incomplete"
            rejection_reasons.append("missing_option_value_lineage")
        elif len(values) > 1:
            status = "conflicted"
            rejection_reasons.append("ambiguous_option_value")
        else:
            status = "supported"
            candidates = next(iter(values.values()))
            path_rank = {"direct": 0, "one_hop": 1, "slot_associated": 2}
            candidates.sort(key=lambda item: (
                path_rank.get(item[1]["path_type"], 3),
                len(item[1].get("raw_value") or ""),
            ))
            selected_value, selected_path = candidates[0]

        intermediate_claim = selected_path.get("intermediate_claim") if selected_path else None
        value_claim = selected_path.get("value_claim") if selected_path else None
        numeric_value_ambiguous = _numeric_value_is_ambiguous(
            value_claim,
            selected_path.get("raw_value") if selected_path else None,
            selected_value,
            question,
        )
        branches.append({
            "option": option,
            "status": status,
            "path_type": selected_path.get("path_type") if selected_path else None,
            "intermediate_entity": selected_path.get("intermediate_entity") if selected_path else None,
            "comparable_value": selected_value,
            "raw_value": selected_path.get("raw_value") if selected_path else None,
            "value_type": selected_value[0] if isinstance(selected_value, tuple) else type(selected_value).__name__ if selected_value is not None else None,
            "value_precision": selected_value[4] if isinstance(selected_value, tuple) and selected_value[0] == 0 else None,
            "intermediate_claim_id": intermediate_claim.get("claim_id") if intermediate_claim else None,
            "value_claim_id": value_claim.get("claim_id") if value_claim else None,
            "claim_ids": [
                str(claim_id) for claim_id in (
                    intermediate_claim.get("claim_id") if intermediate_claim else None,
                    value_claim.get("claim_id") if value_claim else None,
                ) if claim_id
            ],
            "provenance_doc_ids": [
                str(doc_id) for doc_id in (
                    intermediate_claim.get("doc_id") if intermediate_claim else None,
                    value_claim.get("doc_id") if value_claim else None,
                ) if doc_id
            ],
            "candidate_path_count": len(parsed_paths),
            "provenance_entity_compatible": (
                selected_path.get("provenance_entity_compatible") if selected_path else None
            ),
            "provenance_entity_match_status": (
                selected_path.get("provenance_entity_match_status")
                if selected_path else "unknown"
            ),
            "numeric_value_ambiguous": numeric_value_ambiguous,
            "rejection_reasons": rejection_reasons,
        })
    return {
        "intent": intent,
        "branches": branches,
        "build_time_ms": (time.perf_counter() - started) * 1000.0,
    }


def validate_comparison_branches(
    intent: Dict[str, Any],
    branches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reasons = []
    operation = str(intent.get("operation") or "unknown")
    if operation not in {"compare_min", "compare_max", "compare_equal"}:
        reasons.append("comparison_operation_not_supported")
    if operation == "compare_equal":
        reasons.append("compare_equal_requires_canonical_normalization")
    if len(branches) != 2:
        reasons.append("requires_exactly_two_branches")
    if any(branch.get("status") != "supported" for branch in branches):
        reasons.append("incomplete_or_conflicted_branch")
    if len(branches) == 2 and all(branch.get("status") == "supported" for branch in branches):
        value_claim_ids = [str(branch.get("value_claim_id") or "") for branch in branches]
        if not all(value_claim_ids) or len(set(value_claim_ids)) != 2:
            reasons.append("value_claim_reused")
        intermediates = [
            normalize_text(branch.get("intermediate_entity"))
            for branch in branches if branch.get("intermediate_entity")
        ]
        if len(intermediates) == 2 and len(set(intermediates)) != 2:
            reasons.append("intermediate_entity_reused")
        value_types = {str(branch.get("value_type")) for branch in branches}
        if len(value_types) != 1:
            reasons.append("operand_type_mismatch")
        precisions = {
            branch.get("value_precision")
            for branch in branches if branch.get("value_precision") is not None
        }
        if len(precisions) > 1:
            reasons.append("operand_date_precision_mismatch")
    return {
        "status": "executable" if not reasons else "invalid",
        "executable": not reasons,
        "failure_reason": reasons[0] if reasons else None,
        "rejection_reasons": reasons,
    }


def execute_validated_comparison(
    intent: Dict[str, Any],
    branches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    validation = validate_comparison_branches(intent, branches)
    if not validation["executable"]:
        return {**validation, "result": None, "committed_claim_ids": []}
    operation = str(intent.get("operation"))
    values = [branch.get("comparable_value") for branch in branches]
    if operation == "compare_equal":
        result = "yes" if values[0] == values[1] else "no"
    else:
        target = min(values) if operation == "compare_min" else max(values)
        winners = [branch for branch in branches if branch.get("comparable_value") == target]
        if len(winners) != 1:
            return {
                "status": "ambiguous", "executable": False, "result": None,
                "failure_reason": "comparison_tie", "rejection_reasons": ["comparison_tie"],
                "committed_claim_ids": [],
            }
        result = winners[0].get("option")
    return {
        "status": "executed",
        "executable": True,
        "result": result,
        "failure_reason": None,
        "rejection_reasons": [],
        "committed_claim_ids": list(dict.fromkeys(
            claim_id for branch in branches for claim_id in branch.get("claim_ids") or []
        )),
    }


def execute_comparison(
    intent: Dict[str, Any],
    state: Dict[str, Any],
    docs: Optional[List[Dict[str, Any]]] = None,
    question: str = "",
) -> Dict[str, Any]:
    operation = str(intent.get("operation") or "unknown")
    if operation not in {"compare_min", "compare_max", "compare_equal"}:
        return _execute_comparison_strict_slots(intent, state, docs)
    built = build_comparison_branches(question, {**state, "answer_intent": intent}, docs)
    branches = built.get("branches") or []
    execution = execute_validated_comparison(intent, branches)
    if execution.get("executable"):
        return {
            "status": "executed",
            "operator": operation,
            "operands": [
                {
                    "option": branch.get("option"),
                    "value": branch.get("comparable_value"),
                    "claim_ids": branch.get("claim_ids") or [],
                }
                for branch in branches
            ],
            "comparison_lineage": branches,
            "comparison_branches": branches,
            "comparison_branch_status": "executable",
            "comparison_branch_executable": True,
            "comparison_branch_failure_reason": None,
            "comparison_branch_build_time_ms": built.get("build_time_ms", 0.0),
            "operand_binding_status": "valid",
            "strict_comparison_executable": True,
            "result": execution.get("result"),
            "failure_reason": None,
            "rejection_reasons": [],
            "committed_claim_ids": execution.get("committed_claim_ids") or [],
        }
    strict = _execute_comparison_strict_slots(intent, state, docs)
    if strict.get("strict_comparison_executable"):
        branch_control_executable = operation != "compare_equal"
        if not branch_control_executable:
            strict["strict_comparison_executable"] = False
        strict.update({
            "comparison_branches": strict.get("comparison_lineage") or [],
            "comparison_branch_status": "executable" if branch_control_executable else "diagnostic_only",
            "comparison_branch_executable": branch_control_executable,
            "comparison_branch_failure_reason": (
                None if branch_control_executable
                else "compare_equal_requires_canonical_normalization"
            ),
            "comparison_branch_build_time_ms": built.get("build_time_ms", 0.0),
        })
        return strict
    strict.update({
        "comparison_branches": branches,
        "comparison_branch_status": "invalid",
        "comparison_branch_executable": False,
        "comparison_branch_failure_reason": execution.get("failure_reason"),
        "comparison_branch_build_time_ms": built.get("build_time_ms", 0.0),
    })
    return strict


def evaluate_answer_semantics(
    question: str,
    state: Dict[str, Any],
    docs: Optional[List[Dict[str, Any]]] = None,
    candidate_extractor: Optional[Callable[[str, Optional[str], Optional[str]], Optional[str]]] = None,
    answer_type_matcher: Optional[Callable[[str, Optional[str]], bool]] = None,
    enable_comparison_executor: bool = True,
    deterministic_commit_scope: str = "none",
) -> Dict[str, Any]:
    slots = state.get("answer_slots") or []
    intent = build_answer_intent(
        question,
        state.get("question_type"),
        state.get("answer_type"),
        slots,
        state.get("answer_intent"),
    )
    answer_type_matcher = answer_type_matcher or _default_answer_type_match
    title_by_doc_id = _doc_title_map(docs)
    content_by_doc_id = {
        str(doc.get("id", f"doc_{index}")): str(doc.get("contents", doc.get("content", "")))
        for index, doc in enumerate(docs or [])
    }
    resolved = _extract_resolved_entities(state, candidate_extractor, question)
    final_slot_ids = {
        str(slot.get("slot_id"))
        for slot in slots
        if slot.get("slot_role") in {"final_answer", "direct_answer"}
    }
    slots_by_id = {str(slot.get("slot_id")): slot for slot in slots}
    bindings = []

    for claim in state.get("claims") or []:
        claim_id = str(claim.get("claim_id") or "")
        if str(claim.get("slot_id")) not in final_slot_ids or claim.get("support_status") != "support":
            continue
        slot = slots_by_id.get(str(claim.get("slot_id"))) or {}
        anchor_entities = [item.get("entity") for item in resolved if item.get("entity")]
        entity_binding = resolve_claim_binding(claim, anchor_entities, question)
        candidate = str(entity_binding.get("candidate") or "").strip()
        if not candidate and candidate_extractor:
            extracted = candidate_extractor(claim.get("text", ""), intent.get("expected_answer_type"), question) or ""
            if entity_binding.get("entity_binding_valid"):
                candidate = extracted
        subject = str(entity_binding.get("subject") or "")
        haystack = " ".join((subject, str(claim.get("text") or ""), title_by_doc_id.get(str(claim.get("doc_id") or ""), "")))
        requires_subject = intent.get("operation") == "bridge_lookup"
        subject_grounded = bool(entity_binding.get("entity_binding_valid"))
        relation_matched = _relation_matches(intent, claim)
        if (
            intent.get("relation_family") == "capacity"
            and re.search(r"\b(?:can\s+seat|seated|seating)\b", question, re.IGNORECASE)
            and not re.search(
                r"\b(?:seat|seated|seating)\b",
                " ".join((str(claim.get("predicate") or ""), str(claim.get("text") or ""))),
                re.IGNORECASE,
            )
        ):
            relation_matched = False
        doc_id = str(claim.get("doc_id") or "")
        doc_available = not docs or doc_id in title_by_doc_id
        source_content = content_by_doc_id.get(doc_id, "")
        provenance_valid = bool(
            not docs
            or (
                doc_available
                and subject
                and _entity_mentioned(subject, source_content)
                and str(claim.get("object") or "").strip()
                and normalize_text(claim.get("object")) in normalize_text(source_content)
            )
        )
        type_matched = answer_type_matcher(candidate, intent.get("expected_answer_type"))
        candidate_is_intermediate = any(normalize_text(candidate) == item.get("normalized") for item in resolved if candidate)
        candidate_is_question_entity = bool(candidate) and any(
            normalize_text(candidate) == normalize_text(entity)
            for entity in _question_entities(question)
        )
        comparison_option_allowed = intent.get("answer_role") == "option" and any(
            normalize_text(candidate) == normalize_text(option) for option in intent.get("option_entities") or []
        )
        role_valid = bool(candidate) and bool(entity_binding.get("entity_binding_valid")) and not candidate_is_intermediate
        if candidate_is_question_entity and not comparison_option_allowed:
            role_valid = False
        rejection_reasons = []
        if not subject_grounded:
            rejection_reasons.append("subject_not_grounded")
        if not relation_matched:
            rejection_reasons.append("target_relation_mismatch")
        if not type_matched:
            rejection_reasons.append("answer_type_mismatch")
        if not role_valid:
            rejection_reasons.append("candidate_role_invalid")
        if not entity_binding.get("entity_binding_valid"):
            rejection_reasons.append(str(entity_binding.get("entity_binding_failure_reason") or "entity_binding_invalid"))
        if not claim_id:
            rejection_reasons.append("missing_claim_provenance")
        if not doc_available:
            rejection_reasons.append("claim_source_not_in_final_docs")
        elif not provenance_valid:
            rejection_reasons.append("claim_source_does_not_support_endpoints")
        if not claim.get("predicate") or claim.get("object") in (None, ""):
            rejection_reasons.append("missing_structured_claim_semantics")
        if requires_subject and not claim.get("subject"):
            rejection_reasons.append("missing_structured_claim_subject")
        if slot.get("status") == "conflicted" or slot.get("conflicting_claim_ids"):
            rejection_reasons.append("slot_conflicted")
        eligible = not rejection_reasons
        bindings.append({
            "candidate": candidate or None,
            "source": "validated_claim" if eligible else "unvalidated_claim",
            "slot_id": claim.get("slot_id"),
            "claim_id": claim_id or None,
            "subject": subject or None,
            "predicate": claim.get("predicate") or None,
            "object": claim.get("object") or candidate or None,
            "binding_direction": entity_binding.get("binding_direction"),
            "anchor_entity": entity_binding.get("anchor_entity"),
            "anchor_endpoint": entity_binding.get("anchor_endpoint"),
            "candidate_endpoint": entity_binding.get("candidate_endpoint"),
            "entity_binding_valid": bool(entity_binding.get("entity_binding_valid")),
            "entity_binding_failure_reason": entity_binding.get("entity_binding_failure_reason"),
            "subject_grounded": subject_grounded,
            "target_relation_match": relation_matched,
            "candidate_answer_type_match": type_matched,
            "candidate_role_valid": role_valid,
            "claim_source_available": doc_available,
            "provenance_valid": provenance_valid,
            "provenance_doc_id": doc_id if provenance_valid else None,
            "candidate_is_question_entity": candidate_is_question_entity,
            "candidate_is_intermediate_entity": candidate_is_intermediate,
            "eligible": eligible,
            "rejection_reasons": rejection_reasons,
        })

    answer_candidate = str(state.get("answer_candidate") or "").strip()
    answer_candidate_claim_ids = {str(item) for item in state.get("answer_candidate_claim_ids") or []}
    if answer_candidate:
        linked = [item for item in bindings if str(item.get("claim_id")) in answer_candidate_claim_ids]
        eligible = bool(linked) and any(item.get("eligible") for item in linked)
        bindings.append({
            "candidate": answer_candidate,
            "source": "reviewer_answer_candidate",
            "slot_id": linked[0].get("slot_id") if linked else None,
            "claim_id": linked[0].get("claim_id") if linked else None,
            "subject": linked[0].get("subject") if linked else None,
            "predicate": linked[0].get("predicate") if linked else None,
            "object": answer_candidate,
            "binding_direction": linked[0].get("binding_direction") if linked else None,
            "anchor_entity": linked[0].get("anchor_entity") if linked else None,
            "anchor_endpoint": linked[0].get("anchor_endpoint") if linked else None,
            "candidate_endpoint": linked[0].get("candidate_endpoint") if linked else None,
            "entity_binding_valid": bool(linked) and all(item.get("entity_binding_valid") for item in linked),
            "entity_binding_failure_reason": None if eligible else "candidate_claim_not_semantically_valid",
            "subject_grounded": bool(linked) and all(item.get("subject_grounded") for item in linked),
            "target_relation_match": bool(linked) and all(item.get("target_relation_match") for item in linked),
            "candidate_answer_type_match": answer_type_matcher(answer_candidate, intent.get("expected_answer_type")),
            "candidate_role_valid": bool(linked) and all(item.get("candidate_role_valid") for item in linked),
            "candidate_is_question_entity": bool(answer_candidate) and any(
                normalize_text(answer_candidate) == normalize_text(entity)
                for entity in _question_entities(question)
            ),
            "candidate_is_intermediate_entity": any(normalize_text(answer_candidate) == item.get("normalized") for item in resolved),
            "eligible": eligible,
            "rejection_reasons": [] if eligible else ["candidate_claim_not_semantically_valid"],
        })

    comparison = execute_comparison(intent, state, docs, question=question) if enable_comparison_executor else {
        "status": "disabled", "operator": intent.get("operation"), "operands": [], "result": None,
        "comparison_lineage": [], "operand_binding_status": "disabled",
        "strict_comparison_executable": False, "failure_reason": "comparison_executor_disabled",
        "rejection_reasons": [], "committed_claim_ids": [],
    }
    comparison_status = str(comparison.get("status") or "not_attempted")
    derived_answer_status = {
        "executed": "executed",
        "ready": "executable",
        "ambiguous": "ambiguous",
        "incomplete_operands": "invalid",
        "invalid_operands": "invalid",
    }.get(comparison_status, "not_attempted")
    eligible_bindings = [item for item in bindings if item.get("eligible") and item.get("candidate")]
    unique_candidates = {}
    for item in eligible_bindings:
        unique_candidates.setdefault(normalize_text(item["candidate"]), item)

    counterfactual_candidate = None
    counterfactual_source = None
    counterfactual_claim_ids = []
    if comparison.get("status") == "executed":
        counterfactual_candidate = comparison.get("result")
        counterfactual_source = "comparison_executor"
        counterfactual_claim_ids = comparison.get("committed_claim_ids") or []
    elif len(unique_candidates) == 1:
        selected = next(iter(unique_candidates.values()))
        counterfactual_candidate = selected.get("candidate")
        counterfactual_source = selected.get("source")
        counterfactual_claim_ids = [selected.get("claim_id")] if selected.get("claim_id") else []

    strict_comparison_commit = bool(
        deterministic_commit_scope == "strict_comparison"
        and comparison.get("strict_comparison_executable")
        and not intent.get("intent_conflict")
    )
    if strict_comparison_commit:
        answer_decision = {
            "selected_answer_candidate": comparison.get("result"),
            "candidate_source": "comparison_executor",
            "commit_mode": "deterministic_commit",
            "commit_reason": "strict_comparison_lineage_validated",
            "committed_claim_ids": comparison.get("committed_claim_ids") or [],
        }
    else:
        failure_reason = str(state.get("failure_reason") or "")
        allowed_force_reasons = {
            "max_rounds", "parse_failure", "question_analysis_parse_failure",
            "empty_retrieval_plan", "retrieval_budget_exhausted", "no_new_claim",
            "review_failed",
        }
        forced = bool(
            state.get("decision") == "force_answer"
            and (not failure_reason or failure_reason in allowed_force_reasons)
        )
        answer_decision = {
            "selected_answer_candidate": None,
            "candidate_source": None,
            "commit_mode": "forced_generation" if forced else "guided_generation",
            "commit_reason": (
                "original_budget_or_parse_fallback"
                if forced else "strict_comparison_commit_disabled"
                if comparison.get("strict_comparison_executable") else "no_unique_validated_candidate"
                if unique_candidates else "no_validated_candidate"
            ),
            "committed_claim_ids": [],
        }

    has_intermediate = any(slot.get("slot_role") == "intermediate_entity" for slot in slots)
    intermediate_supported = not has_intermediate or bool(resolved)
    final_grounded = bool(eligible_bindings) or comparison.get("status") == "executed"
    if intent.get("operation") in {"compare_min", "compare_max", "compare_equal"}:
        semantic_coverage = 1.0 if comparison.get("status") == "executed" else 0.5 if comparison.get("operands") else 0.0
    elif has_intermediate:
        semantic_coverage = (int(intermediate_supported) + int(final_grounded)) / 2.0
    else:
        semantic_coverage = 1.0 if final_grounded else 0.0

    grounded_ids = [item.get("claim_id") for item in eligible_bindings if item.get("claim_id")]
    all_final_ids = [item.get("claim_id") for item in bindings if item.get("source") != "reviewer_answer_candidate" and item.get("claim_id")]
    chain_trace = {
        "chain_grounded": final_grounded,
        "chain_coverage": semantic_coverage,
        "resolved_intermediate_entities": resolved,
        "grounded_final_claim_ids": list(dict.fromkeys(grounded_ids)),
        "ungrounded_final_claim_ids": [item for item in all_final_ids if item not in grounded_ids],
        "claim_grounding": {
            str(item.get("claim_id")): {
                "grounded": bool(item.get("eligible")),
                "matched_entities": [entry.get("entity") for entry in resolved if item.get("subject_grounded")],
                "reason": "relation_validated" if item.get("eligible") else ",".join(item.get("rejection_reasons") or []),
            }
            for item in bindings
            if item.get("source") != "reviewer_answer_candidate" and item.get("claim_id")
        },
        "ungrounded_reason": None if final_grounded else "no_relation_validated_final_claim",
        "requires_grounding": has_intermediate,
    }
    return {
        "relation_grounded": final_grounded,
        "semantic_coverage": semantic_coverage,
        "answer_intent": intent,
        "resolved_intermediate_entities": resolved,
        "claim_bindings": bindings,
        "comparison_execution": comparison,
        "comparison_execution_status": comparison.get("status"),
        "derived_answer_status": derived_answer_status,
        "comparison_branches": comparison.get("comparison_branches") or [],
        "comparison_branch_status": comparison.get("comparison_branch_status"),
        "comparison_branch_executable": bool(comparison.get("comparison_branch_executable")),
        "comparison_branch_failure_reason": comparison.get("comparison_branch_failure_reason"),
        "comparison_branch_build_time_ms": float(
            comparison.get("comparison_branch_build_time_ms") or 0.0
        ),
        "comparison_lineage": comparison.get("comparison_lineage") or [],
        "operand_binding_status": comparison.get("operand_binding_status"),
        "strict_comparison_executable": bool(comparison.get("strict_comparison_executable")),
        "counterfactual_answer_candidate": counterfactual_candidate,
        "counterfactual_candidate_source": counterfactual_source,
        "counterfactual_claim_ids": counterfactual_claim_ids,
        "would_commit_under_v2_7": bool(counterfactual_candidate),
        "intent_source": intent.get("intent_source"),
        "intent_conflict": bool(intent.get("intent_conflict")),
        "generated_without_validated_candidate": bool(
            answer_decision.get("commit_mode") in {"guided_generation", "forced_generation"}
            and not unique_candidates
        ),
        "answer_decision": answer_decision,
        "subject_grounded": any(item.get("subject_grounded") for item in bindings) if bindings else False,
        "target_relation_match": any(item.get("target_relation_match") for item in bindings) if bindings else False,
        "candidate_role_valid": any(item.get("candidate_role_valid") for item in bindings) if bindings else False,
        "candidate_answer_type_match": any(item.get("candidate_answer_type_match") for item in bindings) if bindings else False,
        "candidate_is_question_entity": any(item.get("candidate_is_question_entity") for item in bindings),
        "candidate_is_intermediate_entity": any(item.get("candidate_is_intermediate_entity") for item in bindings),
        "entity_binding_valid": any(item.get("entity_binding_valid") for item in bindings),
        "binding_directions": list(dict.fromkeys(
            str(item.get("binding_direction"))
            for item in bindings
            if item.get("binding_direction")
        )),
        "grounded_but_wrong_relation": bool(bindings) and any(item.get("subject_grounded") for item in bindings) and not any(item.get("target_relation_match") for item in bindings),
        "chain_grounding_trace": chain_trace,
    }
