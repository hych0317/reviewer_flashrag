"""
Analyze ReviewerPipeline intermediate_data.json files.

The script is read-only. It supports both the legacy prompt-reviewer outputs
and the EvidenceState-controlled outputs.

Usage:
    python3 analyze_reviewer_results.py
    python3 analyze_reviewer_results.py examples/reviewer/result/*/intermediate_data.json
    python3 analyze_reviewer_results.py examples/reviewer/result/*/*/*/intermediate_data.json
    python3 analyze_reviewer_results.py examples/reviewer/result/*/*/*/*/intermediate_data.json
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list in {path}")


def output_of(item: Dict[str, Any]) -> Dict[str, Any]:
    return item.get("output", {})


def metric_of(item: Dict[str, Any]) -> Dict[str, float]:
    return output_of(item).get("metric_score", {}) or {}


def normalized_answer(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def candidate_is_gold(item: Dict[str, Any], candidate: Any) -> bool:
    normalized = normalized_answer(candidate)
    return bool(normalized) and normalized in {
        normalized_answer(answer) for answer in item.get("golden_answers") or []
    }


def diagnostic_of(item: Dict[str, Any]) -> Dict[str, Any]:
    return output_of(item).get("diagnostic_trace", {}) or {}


def has_parse_failure(history: List[Dict[str, Any]], output: Dict[str, Any]) -> bool:
    failure_reason = str(output.get("failure_reason") or "")
    if "parse" in failure_reason:
        return True
    for entry in history or []:
        text = " ".join([
            str(entry.get("failure_reason", "")),
            str(entry.get("explanation", "")),
            str(entry.get("decision", "")),
        ]).lower()
        if "parse" in text or "failed to parse" in text:
            return True
    return False


def has_forced(history: List[Dict[str, Any]], output: Dict[str, Any]) -> bool:
    failure_reason = str(output.get("failure_reason") or "")
    if failure_reason:
        return True
    for entry in history or []:
        decision = str(entry.get("decision", "")).lower()
        iteration = str(entry.get("iteration", "")).lower()
        if "forced" in decision or decision == "force_answer" or iteration == "forced":
            return True
    return False


def has_retrieved_more(history: List[Dict[str, Any]]) -> bool:
    for entry in history or []:
        decision = str(entry.get("decision", "")).lower()
        if decision in {"need_more_info", "continue_retrieval"}:
            return True
        if entry.get("retrieval_plan"):
            return True
    return False


def bucket_name(item: Dict[str, Any]) -> str:
    output = output_of(item)
    history = output.get("review_history") or []
    if has_parse_failure(history, output):
        return "parse_failure"
    if has_forced(history, output):
        return "forced"
    if len(history) == 1 and str(history[0].get("decision", "")).lower() in {"ready_to_answer", "ready"}:
        return "first_ready"
    if has_retrieved_more(history):
        return "retrieved_more_then_ready"
    return "other"


def update_stats(stats: Dict[str, Dict[str, float]], key: str, item: Dict[str, Any]) -> None:
    metric = metric_of(item)
    stats[key]["n"] += 1
    for metric_name in ("em", "f1", "acc", "precision", "recall"):
        stats[key][metric_name] += float(metric.get(metric_name, 0.0))


def print_stats(title: str, stats: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("bucket\tn\tem\tf1\tacc\tprecision\trecall")
    for key in sorted(stats):
        row = stats[key]
        n = int(row["n"])
        if n == 0:
            continue
        values = [row[name] / n for name in ("em", "f1", "acc", "precision", "recall")]
        print(
            f"{key}\t{n}\t"
            + "\t".join(f"{value:.4f}" for value in values)
        )


def gold_recall_bucket(trace: Dict[str, Any]) -> str:
    gold = trace.get("gold_evidence_trace") or {}
    if not gold.get("gold_titles"):
        return "no_gold_trace"
    initial = float(gold.get("initial_gold_title_recall", 0.0))
    final = float(gold.get("final_gold_title_recall", 0.0))
    if initial > 0 and final < initial:
        return "gold_lost_after_ranking"
    if final == 0:
        return "gold_not_found"
    if initial < final:
        return "gold_added_by_adaptive"
    if initial > 0:
        return "gold_found_initial"
    return "gold_other"


def failure_reason_of(item: Dict[str, Any]) -> str:
    output = output_of(item)
    reason = output.get("failure_reason")
    if reason:
        reason = str(reason)
        return reason if len(reason) <= 120 else "textual_failure_reason"
    history = output.get("review_history") or []
    for entry in reversed(history):
        if entry.get("failure_reason"):
            reason = str(entry.get("failure_reason"))
            return reason if len(reason) <= 120 else "textual_failure_reason"
    return "none"


def question_type_of(item: Dict[str, Any]) -> str:
    trace = diagnostic_of(item)
    qa = trace.get("question_analysis") or {}
    if qa.get("question_type"):
        return str(qa.get("question_type"))
    state = output_of(item).get("evidence_state") or {}
    return str(state.get("question_type") or "unknown")


def pipeline_version_of(item: Dict[str, Any]) -> str:
    output = output_of(item)
    trace = diagnostic_of(item)
    pipeline = trace.get("pipeline") or {}
    if pipeline.get("pipeline_version"):
        return str(pipeline.get("pipeline_version"))
    state = output.get("evidence_state") or {}
    return str(state.get("pipeline_version") or "unknown")


def has_missing_slot_with_support_claim(item: Dict[str, Any]) -> bool:
    state = output_of(item).get("evidence_state") or {}
    support_slot_ids = {
        str(claim.get("slot_id"))
        for claim in state.get("claims") or []
        if claim.get("support_status") == "support"
    }
    for slot in state.get("answer_slots") or []:
        if slot.get("status") in {"missing", "partial"} and str(slot.get("slot_id")) in support_slot_ids:
            return True
    return False


def state_consistency_issues(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = diagnostic_of(item)
    issues = []
    for review in trace.get("review_trace") or []:
        issues.extend(review.get("state_consistency_issues") or [])
    state = output_of(item).get("evidence_state") or {}
    issues.extend(state.get("state_consistency_issues") or [])
    return issues


def state_consistency_issue_count(item: Dict[str, Any]) -> int:
    return len(state_consistency_issues(item))


def answer_policy_mismatch(item: Dict[str, Any]) -> bool:
    final_trace = diagnostic_of(item).get("final_trace") or {}
    return bool(final_trace.get("answer_policy_mismatch"))


def chain_grounding_trace_of(item: Dict[str, Any]) -> Dict[str, Any]:
    output = output_of(item)
    trace = diagnostic_of(item)
    final_trace = trace.get("final_trace") or {}
    candidates = [
        trace.get("chain_grounding_trace"),
        final_trace.get("chain_grounding_trace"),
        (output.get("adoption_policy") or {}).get("chain_grounding_trace"),
        (output.get("evidence_state") or {}).get("chain_grounding_trace"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def relation_grounding_trace_of(item: Dict[str, Any]) -> Dict[str, Any]:
    output = output_of(item)
    trace = diagnostic_of(item)
    final_trace = trace.get("final_trace") or {}
    candidates = [
        trace.get("relation_grounding_trace"),
        final_trace.get("relation_grounding_trace"),
        (output.get("adoption_policy") or {}).get("relation_grounding_trace"),
        (output.get("evidence_state") or {}).get("relation_grounding_trace"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def answer_decision_of(item: Dict[str, Any]) -> Dict[str, Any]:
    output = output_of(item)
    trace = diagnostic_of(item)
    final_trace = trace.get("final_trace") or {}
    for candidate in (
        trace.get("answer_decision"),
        final_trace.get("answer_decision"),
        (output.get("adoption_policy") or {}).get("answer_decision"),
        (output.get("evidence_state") or {}).get("answer_decision"),
    ):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def consistency_issue_repaired(issue: Dict[str, Any]) -> bool:
    if "repaired" in issue:
        return bool(issue.get("repaired"))
    return str(issue.get("reason") or "") in {
        "missing_slot_has_support_claim",
        "support_and_contradict_claims",
    }


def consistency_issue_blocked(issue: Dict[str, Any]) -> bool:
    return bool(issue.get("blocked_reason"))


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}")
    if not counter:
        print("(empty)")
        return
    for key, value in counter.most_common():
        print(f"{key}\t{value}")


def print_retrieval_summary(items: List[Dict[str, Any]]) -> None:
    traced = 0
    total_queries = 0
    total_new_docs = 0
    adaptive_gold_hits = 0
    slot_statuses = Counter()
    answer_types = Counter()
    repair_methods = Counter()
    candidate_sources = Counter()
    candidate_source_by_chain_grounding = Counter()
    pipeline_versions = Counter()
    state_schema_versions = Counter()
    guard_extra_used = 0
    guard_triggered = 0
    answer_supported_count = 0
    answer_unsupported_count = 0
    verifier_enabled_count = 0
    coverage_sum = 0.0
    coverage_count = 0
    case_candidates = Counter()
    high_coverage_wrong_by_answer_type = Counter()
    missing_slot_has_support_claim = 0
    state_consistency_repaired = 0
    policy_mismatch = 0
    policy_mismatch_wrong = 0
    state_consistency_observed = 0
    state_consistency_blocked = 0
    state_consistency_blocked_reasons = Counter()
    chain_trace_count = 0
    chain_grounded_count = 0
    chain_coverage_sum = 0.0
    chain_coverage_count = 0
    ungrounded_high_coverage_wrong = 0
    max_rounds_null_wrong = 0
    relation_trace_count = 0
    relation_grounded_count = 0
    semantic_coverage_sum = 0.0
    semantic_ready_wrong = 0
    grounded_but_relation_mismatch = 0
    comparison_executed = 0
    comparison_executed_correct = 0
    deterministic_commit = 0
    deterministic_commit_correct = 0
    policy_mismatch_by_commit_mode = Counter()
    high_coverage_wrong_by_operation = Counter()
    candidate_source_by_operation = Counter()
    control_actions = Counter()
    intent_sources = Counter()
    operand_binding_statuses = Counter()
    intent_conflicts = 0
    strict_comparison_executable = 0
    strict_comparison_correct = 0
    counterfactual_candidates = 0
    counterfactual_correct = 0
    would_commit_under_v2_7 = 0
    forced_by_relation_guard = 0
    generated_without_validated_candidate = 0
    question_type_repaired = 0
    entity_binding_valid = 0
    diagnostic_candidate_leakage = 0
    semantic_guidance_applied = 0
    provenance_rebind_candidates = 0
    date_precision_mismatch = 0
    binding_directions = Counter()
    comparison_items = 0
    comparison_operand_complete = 0
    comparison_extra_final_slot = 0
    comparison_forced_with_complete_operands = 0
    comparison_ready_with_complete_operands = 0
    comparison_lineage_crossed = 0
    comparison_invalid_reasons = Counter()
    diagnostic_behavior_mutations = 0
    comparison_view_applied = 0
    derived_slots_hidden_from_reviewer = 0
    derived_retrieval_plans_dropped = 0
    non_comparison_prompt_mutated = 0
    non_comparison_query_mutated = 0
    non_comparison_output_mutated = 0
    comparison_branch_executable = 0
    comparison_branch_executable_correct = 0
    comparison_branch_executable_wrong = 0
    comparison_branch_fallback = 0
    comparison_branch_fallback_correct = 0
    comparison_guidance_applied = 0
    comparison_branch_build_time_ms = 0.0
    comparison_branch_failure_reasons = Counter()
    comparison_branch_by_operation = Counter()
    comparison_option_or_intermediate_output = 0
    comparison_output_contract_available = 0
    comparison_output_contract_applied = 0
    comparison_commit_eligible = 0
    comparison_commit_eligible_correct = 0
    comparison_final_commits = 0
    comparison_final_commit_correct = 0
    comparison_slot_associated_rejected = 0
    comparison_guided_generation = 0
    comparison_saved_generation = 0
    comparison_commit_blocked_reasons = Counter()
    comparison_finalization_modes = Counter()
    comparison_commit_confidences = Counter()
    comparison_commit_relation_families = Counter()
    comparison_commit_relation_value_types = Counter()
    provenance_entity_match_statuses = Counter()

    for item in items:
        trace = diagnostic_of(item)
        if not trace:
            continue
        traced += 1
        pipeline = trace.get("pipeline") or {}
        pipeline_versions[str(pipeline.get("pipeline_version") or "unknown")] += 1
        retrieval_trace = trace.get("retrieval_trace") or []
        total_queries += len(retrieval_trace)
        total_new_docs += sum(int(entry.get("new_doc_count") or 0) for entry in retrieval_trace)

        gold = trace.get("gold_evidence_trace") or {}
        normalized_gold = {str(title).strip().lower() for title in gold.get("gold_titles") or []}
        for entry in retrieval_trace:
            titles = {str(title).strip().lower() for title in entry.get("retrieved_doc_titles") or []}
            if normalized_gold and titles.intersection(normalized_gold):
                adaptive_gold_hits += 1

        state = output_of(item).get("evidence_state") or {}
        final_trace = trace.get("final_trace") or {}
        commit_eligible = False
        final_commit = False
        question_type_repaired += int(bool(
            final_trace.get("question_type_repaired") or state.get("question_type_repaired")
        ))
        state_schema_versions[str(state.get("state_schema_version") or "unknown")] += 1
        answer_type = state.get("answer_type")
        if answer_type:
            answer_types[str(answer_type)] += 1
        if state.get("ready_guard_triggered"):
            guard_triggered += 1
        if state.get("guard_extra_round_used"):
            guard_extra_used += 1
        if has_missing_slot_with_support_claim(item):
            missing_slot_has_support_claim += 1
        consistency_issues = state_consistency_issues(item)
        if consistency_issues:
            state_consistency_observed += 1
        if any(consistency_issue_repaired(issue) for issue in consistency_issues):
            state_consistency_repaired += 1
        if any(consistency_issue_blocked(issue) for issue in consistency_issues):
            state_consistency_blocked += 1
        for issue in consistency_issues:
            if issue.get("blocked_reason"):
                state_consistency_blocked_reasons[str(issue.get("blocked_reason"))] += 1

        for slot in state.get("answer_slots") or []:
            slot_statuses[str(slot.get("status") or "unknown")] += 1

        if state.get("question_type") == "comparison":
            comparison_items += 1
            slots = state.get("answer_slots") or []
            operands = [slot for slot in slots if slot.get("slot_role") == "comparison_operand"]
            operands_complete = bool(len(operands) >= 2) and all(
                slot.get("status") == "supported" for slot in operands
            )
            comparison_operand_complete += int(operands_complete)
            comparison_extra_final_slot += int(any(
                slot.get("slot_role") == "final_answer"
                and not slot.get("derived_output")
                for slot in slots
            ))
            decision = str(state.get("decision") or "")
            comparison_forced_with_complete_operands += int(
                operands_complete and decision == "force_answer"
            )
            comparison_ready_with_complete_operands += int(
                operands_complete and decision == "ready_to_answer"
            )
            comparison_view_applied += int(bool(
                final_trace.get("comparison_view_applied")
                or state.get("comparison_view_applied")
            ))
            derived_slots_hidden_from_reviewer += int(
                final_trace.get("derived_slots_hidden_from_reviewer")
                or state.get("derived_slots_hidden_from_reviewer")
                or 0
            )
            derived_retrieval_plans_dropped += int(
                final_trace.get("derived_retrieval_plans_dropped")
                or state.get("derived_retrieval_plans_dropped")
                or 0
            )
            output_contract_applied = bool(
                final_trace.get("comparison_output_contract_applied")
                or trace.get("comparison_output_contract_applied")
            )
            output_contract_available = bool(
                final_trace.get("comparison_output_contract_available")
                or trace.get("comparison_output_contract_available")
                or final_trace.get("comparison_output_contract")
                or trace.get("comparison_output_contract")
            )
            commit_eligible = bool(
                final_trace.get("comparison_commit_eligible")
                or trace.get("comparison_commit_eligible")
            )
            final_commit = bool(
                final_trace.get("comparison_final_commit_saved_generation")
                or trace.get("comparison_final_commit_saved_generation")
            )
            comparison_output_contract_available += int(output_contract_available)
            comparison_output_contract_applied += int(output_contract_applied)
            comparison_commit_eligible += int(commit_eligible)
            comparison_final_commits += int(final_commit)
            comparison_slot_associated_rejected += int(bool(
                final_trace.get("comparison_slot_associated_rejected")
                or trace.get("comparison_slot_associated_rejected")
            ))
            comparison_guided_generation += int(bool(
                final_trace.get("comparison_guided_generation")
                or trace.get("comparison_guided_generation")
            ))
            comparison_saved_generation += int(final_commit)
            blocked_reason = (
                final_trace.get("comparison_commit_blocked_reason")
                or trace.get("comparison_commit_blocked_reason")
            )
            if blocked_reason:
                comparison_commit_blocked_reasons[str(blocked_reason)] += 1
            commit_confidence = (
                final_trace.get("comparison_commit_confidence")
                or trace.get("comparison_commit_confidence")
            )
            if commit_confidence:
                comparison_commit_confidences[str(commit_confidence)] += 1
            commit_relation_family = (
                final_trace.get("comparison_commit_relation_family")
                or trace.get("comparison_commit_relation_family")
            )
            if commit_relation_family:
                comparison_commit_relation_families[str(commit_relation_family)] += 1
            relation_value_type_valid = (
                final_trace.get("comparison_commit_relation_value_type_valid")
                if "comparison_commit_relation_value_type_valid" in final_trace
                else trace.get("comparison_commit_relation_value_type_valid")
            )
            if relation_value_type_valid is not None:
                comparison_commit_relation_value_types[
                    "valid" if relation_value_type_valid else "invalid"
                ] += 1
            finalization_mode = (
                final_trace.get("comparison_finalization_mode")
                or trace.get("comparison_finalization_mode")
                or "unknown"
            )
            comparison_finalization_modes[str(finalization_mode)] += 1
        else:
            non_comparison_prompt_mutated += int(bool(
                final_trace.get("non_comparison_prompt_mutated")
            ))
            non_comparison_query_mutated += int(bool(
                final_trace.get("non_comparison_query_mutated")
            ))
            non_comparison_output_mutated += int(bool(
                final_trace.get("non_comparison_output_mutated")
            ))

        policy = output_of(item).get("adoption_policy") or {}
        candidate_source = str(policy.get("candidate_source") or "null")
        if policy.get("candidate_source"):
            candidate_sources[candidate_source] += 1
        chain_trace = chain_grounding_trace_of(item)
        if chain_trace:
            chain_trace_count += 1
            chain_grounded = bool(chain_trace.get("chain_grounded"))
            if chain_grounded:
                chain_grounded_count += 1
            if "chain_coverage" in chain_trace:
                chain_coverage_sum += float(chain_trace.get("chain_coverage") or 0.0)
                chain_coverage_count += 1
            chain_bucket = "grounded" if chain_grounded else "ungrounded"
        else:
            chain_bucket = "unknown"
        candidate_source_by_chain_grounding[f"{chain_bucket}:{candidate_source}"] += 1

        relation_trace = relation_grounding_trace_of(item)
        answer_decision = answer_decision_of(item)
        operation = str(
            (relation_trace.get("answer_intent") or {}).get("operation")
            or ((state.get("answer_intent") or {}).get("operation"))
            or "unknown"
        )
        commit_mode = str(answer_decision.get("commit_mode") or "unknown")
        candidate_source_by_operation[f"{operation}:{candidate_source}"] += 1
        if relation_trace:
            relation_trace_count += 1
            if relation_trace.get("relation_grounded"):
                relation_grounded_count += 1
            semantic_coverage_sum += float(relation_trace.get("semantic_coverage") or 0.0)
            if relation_trace.get("grounded_but_wrong_relation"):
                grounded_but_relation_mismatch += 1
            control_actions[str(relation_trace.get("control_action") or "unknown")] += 1
            intent_sources[str(relation_trace.get("intent_source") or "unknown")] += 1
            operand_binding_statuses[str(relation_trace.get("operand_binding_status") or "unknown")] += 1
            intent_conflicts += int(bool(relation_trace.get("intent_conflict")))
            strict = bool(relation_trace.get("strict_comparison_executable"))
            strict_comparison_executable += int(strict)
            for branch in relation_trace.get("comparison_branches") or []:
                provenance_entity_match_statuses[str(
                    branch.get("provenance_entity_match_status") or "unknown"
                )] += 1
            counterfactual = relation_trace.get("counterfactual_answer_candidate")
            if counterfactual:
                counterfactual_candidates += 1
                counterfactual_correct += int(candidate_is_gold(item, counterfactual))
                strict_comparison_correct += int(strict and candidate_is_gold(item, counterfactual))
            would_commit_under_v2_7 += int(bool(relation_trace.get("would_commit_under_v2_7")))
            final_trace = trace.get("final_trace") or {}
            forced_by_relation_guard += int(bool(
                relation_trace.get("forced_by_relation_guard")
                or final_trace.get("forced_by_relation_guard")
                or trace.get("forced_by_relation_guard")
            ))
            generated_without_validated_candidate += int(bool(
                relation_trace.get("generated_without_validated_candidate")
                or final_trace.get("generated_without_validated_candidate")
                or trace.get("generated_without_validated_candidate")
            ))
            entity_binding_valid += int(bool(relation_trace.get("entity_binding_valid")))
            for direction in relation_trace.get("binding_directions") or []:
                binding_directions[str(direction)] += 1
            diagnostic_candidate_leakage += int(bool(
                final_trace.get("diagnostic_candidate_leakage")
                or trace.get("diagnostic_candidate_leakage")
            ))
            semantic_guidance_applied += int(bool(
                final_trace.get("semantic_guidance_applied")
                or trace.get("semantic_guidance_applied")
            ))
            comparison = relation_trace.get("comparison_execution") or {}
            branch_executable = bool(relation_trace.get("comparison_branch_executable"))
            comparison_branch_executable += int(branch_executable)
            comparison_branch_build_time_ms += float(
                relation_trace.get("comparison_branch_build_time_ms") or 0.0
            )
            comparison_branch_by_operation[
                f"{operation}:{'executable' if branch_executable else 'fallback'}"
            ] += 1
            if relation_trace.get("comparison_branch_failure_reason"):
                comparison_branch_failure_reasons[str(
                    relation_trace.get("comparison_branch_failure_reason")
                )] += 1
            diagnostic_behavior_mutations += int(any(
                relation_trace.get(name)
                for name in (
                    "diagnostic_state_mutated", "diagnostic_query_mutated",
                    "diagnostic_prompt_mutated", "diagnostic_answer_mutated",
                )
            ))
            comparison_lineage_crossed += int(
                "operand_lineage_crossed" in (comparison.get("rejection_reasons") or [])
            )
            if comparison.get("failure_reason"):
                comparison_invalid_reasons[str(comparison.get("failure_reason"))] += 1
            for lineage in comparison.get("comparison_lineage") or []:
                provenance_rebind_candidates += int(bool(
                    lineage.get("alternative_intermediate_provenance_doc_ids")
                    or lineage.get("alternative_value_provenance_doc_ids")
                ))
            date_precision_mismatch += int(
                "operand_date_precision_mismatch" in (comparison.get("rejection_reasons") or [])
            )

        verifier = output_of(item).get("verifier_trace") or trace.get("verifier_trace") or {}
        if verifier.get("enabled"):
            verifier_enabled_count += 1
        answer_supported = output_of(item).get("answer_supported")
        if answer_supported is None and "answer_supported" in verifier:
            answer_supported = verifier.get("answer_supported")
        if answer_supported is True:
            answer_supported_count += 1
        elif answer_supported is False:
            answer_unsupported_count += 1

        qa = trace.get("question_analysis") or {}
        if qa.get("repair_method"):
            repair_methods[str(qa.get("repair_method"))] += 1
        for review in trace.get("review_trace") or []:
            if review.get("repair_method"):
                repair_methods[str(review.get("repair_method"))] += 1

        coverage = output_of(item).get("slot_coverage") or {}
        if "coverage" in coverage:
            coverage_sum += float(coverage.get("coverage") or 0.0)
            coverage_count += 1

        metric = metric_of(item)
        bucket = gold_recall_bucket(trace)
        forced = has_forced(output_of(item).get("review_history") or [], output_of(item))
        f1 = float(metric.get("f1", 0.0))
        if state.get("question_type") == "comparison":
            comparison_candidate = (
                (final_trace.get("answer_decision") or {}).get("selected_answer_candidate")
                or (output_of(item).get("adoption_policy") or {}).get("selected_answer_candidate")
                or relation_trace.get("counterfactual_answer_candidate")
            )
            comparison_candidate_correct = candidate_is_gold(item, comparison_candidate)
            comparison_commit_eligible_correct += int(
                commit_eligible and comparison_candidate_correct
            )
            comparison_final_commit_correct += int(
                final_commit and comparison_candidate_correct
            )
        if state.get("question_type") == "comparison":
            branch_executable = bool(relation_trace.get("comparison_branch_executable"))
            branch_candidate_correct = candidate_is_gold(
                item, relation_trace.get("counterfactual_answer_candidate")
            )
            comparison_branch_executable_correct += int(
                branch_executable and branch_candidate_correct
            )
            comparison_branch_executable_wrong += int(
                branch_executable and not branch_candidate_correct
            )
            fallback = bool(
                final_trace.get("comparison_fallback_to_patch2")
                or state.get("comparison_fallback_to_patch2")
            )
            comparison_branch_fallback += int(fallback)
            comparison_branch_fallback_correct += int(fallback and f1 > 0)
            comparison_guidance_applied += int(bool(
                final_trace.get("comparison_guidance_applied")
            ))
            pred_norm = normalized_answer(str(output_of(item).get("pred") or ""))
            options = [
                normalized_answer(str(option))
                for option in ((state.get("answer_intent") or {}).get("option_entities") or [])
            ]
            intermediates = [
                normalized_answer(str(branch.get("intermediate_entity") or ""))
                for branch in relation_trace.get("comparison_branches") or []
                if branch.get("intermediate_entity")
            ]
            comparison_option_or_intermediate_output += int(bool(
                pred_norm and pred_norm in set(options + intermediates) and f1 == 0
            ))
        if answer_policy_mismatch(item):
            policy_mismatch += 1
            policy_mismatch_by_commit_mode[commit_mode] += 1
            if f1 == 0:
                policy_mismatch_wrong += 1
        if relation_trace.get("relation_grounded") and float(relation_trace.get("semantic_coverage") or 0.0) >= 0.99 and f1 == 0:
            semantic_ready_wrong += 1
        comparison_status = str(relation_trace.get("comparison_execution_status") or "")
        if comparison_status == "executed":
            comparison_executed += 1
            comparison_executed_correct += int(f1 > 0)
        if commit_mode == "deterministic_commit":
            deterministic_commit += 1
            deterministic_commit_correct += int(f1 > 0)
        if bucket == "gold_added_by_adaptive" and f1 > 0:
            case_candidates["adaptive_added_gold_answered"] += 1
        if gold.get("final_gold_title_recall", 0.0) > 0 and f1 == 0:
            case_candidates["retrieved_gold_but_wrong"] += 1
        if coverage.get("coverage", 0.0) >= 0.99 and f1 == 0:
            case_candidates["high_coverage_but_wrong"] += 1
            high_coverage_wrong_by_answer_type[str(answer_type or "unknown")] += 1
            high_coverage_wrong_by_operation[operation] += 1
            if chain_trace and not chain_trace.get("chain_grounded"):
                ungrounded_high_coverage_wrong += 1
                case_candidates["high_coverage_ungrounded_wrong"] += 1
        if forced and bucket == "gold_not_found":
            case_candidates["forced_gold_missing"] += 1
        if gold.get("final_gold_title_recall", 0.0) > 0 and f1 == 0 and chain_trace and not chain_trace.get("chain_grounded"):
            case_candidates["retrieved_gold_ungrounded_wrong"] += 1
        if failure_reason_of(item) == "max_rounds" and candidate_source == "null" and f1 == 0:
            max_rounds_null_wrong += 1
            case_candidates["max_rounds_null_wrong"] += 1

    print("\ndiagnostic_summary")
    print(f"traced_items\t{traced}")
    if traced:
        print(f"avg_retrieval_queries\t{total_queries / traced:.3f}")
        print(f"avg_new_docs\t{total_new_docs / traced:.3f}")
        print(f"adaptive_query_gold_hit_rate\t{adaptive_gold_hits / total_queries:.4f}" if total_queries else "adaptive_query_gold_hit_rate\t0.0000")
        print(f"avg_slot_coverage\t{coverage_sum / coverage_count:.4f}" if coverage_count else "avg_slot_coverage\t0.0000")
        print(f"guard_trigger_rate\t{guard_triggered / traced:.4f}")
        print(f"guard_extra_used_rate\t{guard_extra_used / traced:.4f}")
        print(f"missing_slot_has_support_claim\t{missing_slot_has_support_claim}")
        print(f"state_consistency_observed_rate\t{state_consistency_observed / traced:.4f}")
        print(f"state_consistency_repaired_rate\t{state_consistency_repaired / traced:.4f}")
        print(f"state_consistency_blocked_rate\t{state_consistency_blocked / traced:.4f}")
        print(f"policy_mismatch_rate\t{policy_mismatch / traced:.4f}")
        print(f"policy_mismatch_wrong\t{policy_mismatch_wrong}")
        print(f"chain_grounded_rate\t{chain_grounded_count / chain_trace_count:.4f}" if chain_trace_count else "chain_grounded_rate\t0.0000")
        print(f"chain_coverage_avg\t{chain_coverage_sum / chain_coverage_count:.4f}" if chain_coverage_count else "chain_coverage_avg\t0.0000")
        print(f"ungrounded_high_coverage_wrong\t{ungrounded_high_coverage_wrong}")
        print(f"max_rounds_null_wrong\t{max_rounds_null_wrong}")
        print(f"relation_grounded_rate\t{relation_grounded_count / relation_trace_count:.4f}" if relation_trace_count else "relation_grounded_rate\t0.0000")
        print(f"semantic_coverage_avg\t{semantic_coverage_sum / relation_trace_count:.4f}" if relation_trace_count else "semantic_coverage_avg\t0.0000")
        print(f"semantic_ready_wrong\t{semantic_ready_wrong}")
        print(f"grounded_but_relation_mismatch\t{grounded_but_relation_mismatch}")
        print(f"comparison_executor_coverage\t{comparison_executed / traced:.4f}")
        print(f"comparison_executor_accuracy\t{comparison_executed_correct / comparison_executed:.4f}" if comparison_executed else "comparison_executor_accuracy\t0.0000")
        print(f"deterministic_commit_rate\t{deterministic_commit / traced:.4f}")
        print(f"deterministic_commit_accuracy\t{deterministic_commit_correct / deterministic_commit:.4f}" if deterministic_commit else "deterministic_commit_accuracy\t0.0000")
        print(f"intent_conflict_rate\t{intent_conflicts / traced:.4f}")
        print(f"strict_comparison_executable_rate\t{strict_comparison_executable / traced:.4f}")
        print(f"strict_comparison_counterfactual_accuracy\t{strict_comparison_correct / strict_comparison_executable:.4f}" if strict_comparison_executable else "strict_comparison_counterfactual_accuracy\t0.0000")
        print(f"counterfactual_candidate_rate\t{counterfactual_candidates / traced:.4f}")
        print(f"counterfactual_candidate_accuracy\t{counterfactual_correct / counterfactual_candidates:.4f}" if counterfactual_candidates else "counterfactual_candidate_accuracy\t0.0000")
        print(f"would_commit_under_v2_7_rate\t{would_commit_under_v2_7 / traced:.4f}")
        print(f"forced_by_relation_guard_rate\t{forced_by_relation_guard / traced:.4f}")
        print(f"generated_without_validated_candidate_rate\t{generated_without_validated_candidate / traced:.4f}")
        print(f"question_type_repaired_rate\t{question_type_repaired / traced:.4f}")
        print(f"entity_binding_valid_rate\t{entity_binding_valid / traced:.4f}")
        print(f"diagnostic_candidate_leakage_rate\t{diagnostic_candidate_leakage / traced:.4f}")
        print(f"semantic_guidance_applied_rate\t{semantic_guidance_applied / traced:.4f}")
        print(f"provenance_rebind_candidates\t{provenance_rebind_candidates}")
        print(f"date_precision_mismatch\t{date_precision_mismatch}")
        print(f"comparison_operand_complete_rate\t{comparison_operand_complete / comparison_items:.4f}" if comparison_items else "comparison_operand_complete_rate\t0.0000")
        print(f"comparison_extra_final_slot_rate\t{comparison_extra_final_slot / comparison_items:.4f}" if comparison_items else "comparison_extra_final_slot_rate\t0.0000")
        print(f"comparison_ready_with_complete_operands\t{comparison_ready_with_complete_operands}")
        print(f"comparison_forced_with_complete_operands\t{comparison_forced_with_complete_operands}")
        print(f"comparison_lineage_crossed\t{comparison_lineage_crossed}")
        print(f"diagnostic_behavior_mutation_rate\t{diagnostic_behavior_mutations / relation_trace_count:.4f}" if relation_trace_count else "diagnostic_behavior_mutation_rate\t0.0000")
        print(f"comparison_view_applied_rate\t{comparison_view_applied / comparison_items:.4f}" if comparison_items else "comparison_view_applied_rate\t0.0000")
        print(f"derived_slots_hidden_from_reviewer\t{derived_slots_hidden_from_reviewer}")
        print(f"derived_retrieval_plans_dropped\t{derived_retrieval_plans_dropped}")
        non_comparison_items = traced - comparison_items
        print(f"non_comparison_prompt_mutation_rate\t{non_comparison_prompt_mutated / non_comparison_items:.4f}" if non_comparison_items else "non_comparison_prompt_mutation_rate\t0.0000")
        print(f"non_comparison_query_mutation_rate\t{non_comparison_query_mutated / non_comparison_items:.4f}" if non_comparison_items else "non_comparison_query_mutation_rate\t0.0000")
        print(f"non_comparison_output_mutation_rate\t{non_comparison_output_mutated / non_comparison_items:.4f}" if non_comparison_items else "non_comparison_output_mutation_rate\t0.0000")
        print(f"comparison_branch_executable_coverage\t{comparison_branch_executable / comparison_items:.4f}" if comparison_items else "comparison_branch_executable_coverage\t0.0000")
        print(f"comparison_branch_executable_accuracy\t{comparison_branch_executable_correct / comparison_branch_executable:.4f}" if comparison_branch_executable else "comparison_branch_executable_accuracy\t0.0000")
        print(f"comparison_branch_executable_wrong\t{comparison_branch_executable_wrong}")
        print(f"comparison_branch_fallback_rate\t{comparison_branch_fallback / comparison_items:.4f}" if comparison_items else "comparison_branch_fallback_rate\t0.0000")
        print(f"comparison_branch_fallback_accuracy\t{comparison_branch_fallback_correct / comparison_branch_fallback:.4f}" if comparison_branch_fallback else "comparison_branch_fallback_accuracy\t0.0000")
        print(f"comparison_guidance_applied_rate\t{comparison_guidance_applied / comparison_items:.4f}" if comparison_items else "comparison_guidance_applied_rate\t0.0000")
        print(f"comparison_branch_build_time_ms_avg\t{comparison_branch_build_time_ms / comparison_items:.4f}" if comparison_items else "comparison_branch_build_time_ms_avg\t0.0000")
        print(f"comparison_option_or_intermediate_output\t{comparison_option_or_intermediate_output}")
        print(f"comparison_output_contract_available_rate\t{comparison_output_contract_available / comparison_items:.4f}" if comparison_items else "comparison_output_contract_available_rate\t0.0000")
        print(f"comparison_output_contract_coverage\t{comparison_output_contract_applied / comparison_items:.4f}" if comparison_items else "comparison_output_contract_coverage\t0.0000")
        print(f"comparison_final_commit_eligible_rate\t{comparison_commit_eligible / comparison_items:.4f}" if comparison_items else "comparison_final_commit_eligible_rate\t0.0000")
        print(f"comparison_final_commit_eligible_accuracy\t{comparison_commit_eligible_correct / comparison_commit_eligible:.4f}" if comparison_commit_eligible else "comparison_final_commit_eligible_accuracy\t0.0000")
        print(f"comparison_final_commit_rate\t{comparison_final_commits / comparison_items:.4f}" if comparison_items else "comparison_final_commit_rate\t0.0000")
        print(f"comparison_final_commit_accuracy\t{comparison_final_commit_correct / comparison_final_commits:.4f}" if comparison_final_commits else "comparison_final_commit_accuracy\t0.0000")
        print(f"comparison_guided_generation_rate\t{comparison_guided_generation / comparison_items:.4f}" if comparison_items else "comparison_guided_generation_rate\t0.0000")
        print(f"comparison_final_commit_saved_generation\t{comparison_saved_generation}")
        print(f"comparison_slot_associated_rejected\t{comparison_slot_associated_rejected}")
        total_answer_support = answer_supported_count + answer_unsupported_count
        print(f"answer_unsupported_rate\t{answer_unsupported_count / total_answer_support:.4f}" if total_answer_support else "answer_unsupported_rate\t0.0000")
        print(f"verifier_enabled_rate\t{verifier_enabled_count / traced:.4f}")
    print_counter("pipeline_versions", pipeline_versions)
    print_counter("state_schema_versions", state_schema_versions)
    print_counter("slot_statuses", slot_statuses)
    print_counter("answer_types", answer_types)
    print_counter("repair_methods", repair_methods)
    print_counter("candidate_sources", candidate_sources)
    print_counter("candidate_source_by_chain_grounding", candidate_source_by_chain_grounding)
    print_counter("candidate_source_by_operation", candidate_source_by_operation)
    print_counter("control_actions", control_actions)
    print_counter("intent_sources", intent_sources)
    print_counter("operand_binding_statuses", operand_binding_statuses)
    print_counter("binding_directions", binding_directions)
    print_counter("comparison_invalid_reasons", comparison_invalid_reasons)
    print_counter("comparison_branch_failure_reasons", comparison_branch_failure_reasons)
    print_counter("comparison_branch_by_operation", comparison_branch_by_operation)
    print_counter("comparison_commit_blocked_reasons", comparison_commit_blocked_reasons)
    print_counter("comparison_finalization_modes", comparison_finalization_modes)
    print_counter("comparison_commit_confidences", comparison_commit_confidences)
    print_counter("comparison_commit_relation_families", comparison_commit_relation_families)
    print_counter(
        "comparison_commit_relation_value_types",
        comparison_commit_relation_value_types,
    )
    print_counter("provenance_entity_match_statuses", provenance_entity_match_statuses)
    print_counter("policy_mismatch_by_commit_mode", policy_mismatch_by_commit_mode)
    print_counter("state_consistency_blocked_reasons", state_consistency_blocked_reasons)
    print_counter("high_coverage_wrong_by_answer_type", high_coverage_wrong_by_answer_type)
    print_counter("high_coverage_wrong_by_operation", high_coverage_wrong_by_operation)
    print_counter("case_study_candidates", case_candidates)


def analyze_file(path: str) -> None:
    items = load_items(path)
    overall = defaultdict(lambda: defaultdict(float))
    by_bucket = defaultdict(lambda: defaultdict(float))
    by_type = defaultdict(lambda: defaultdict(float))
    by_gold_recall_change = defaultdict(lambda: defaultdict(float))
    by_question_type = defaultdict(lambda: defaultdict(float))
    by_failure_reason = defaultdict(lambda: defaultdict(float))
    by_pipeline_version = defaultdict(lambda: defaultdict(float))
    decisions = Counter()
    forced = 0
    parse_failure = 0
    histories = 0
    history_len = 0

    for item in items:
        output = output_of(item)
        history = output.get("review_history") or []
        histories += 1 if history else 0
        history_len += len(history)
        for entry in history:
            decisions[str(entry.get("decision"))] += 1

        if has_forced(history, output):
            forced += 1
        if has_parse_failure(history, output):
            parse_failure += 1

        update_stats(overall, "all", item)
        update_stats(by_bucket, bucket_name(item), item)
        update_stats(by_gold_recall_change, gold_recall_bucket(diagnostic_of(item)), item)
        update_stats(by_question_type, question_type_of(item), item)
        update_stats(by_failure_reason, failure_reason_of(item), item)
        update_stats(by_pipeline_version, pipeline_version_of(item), item)
        sample_type = (item.get("metadata") or {}).get("type")
        if sample_type:
            update_stats(by_type, sample_type, item)

    print(f"\n{'=' * 80}")
    print(path)
    print(f"items: {len(items)}")
    print(f"with_history: {histories}")
    print(f"avg_history_len: {history_len / histories:.3f}" if histories else "avg_history_len: 0.000")
    print(f"forced_items: {forced}")
    print(f"forced_rate: {forced / len(items):.4f}" if items else "forced_rate: 0.0000")
    print(f"parse_failure_items: {parse_failure}")
    print(f"parse_failure_rate: {parse_failure / len(items):.4f}" if items else "parse_failure_rate: 0.0000")
    print(f"decisions: {dict(decisions)}")
    print_stats("overall", overall)
    print_stats("by_bucket", by_bucket)
    if by_type:
        print_stats("by_hotpot_type", by_type)
    print_stats("by_gold_recall_change", by_gold_recall_change)
    print_stats("by_question_type", by_question_type)
    print_stats("by_failure_reason", by_failure_reason)
    print_stats("by_pipeline_version", by_pipeline_version)
    print_retrieval_summary(items)


def expand_paths(patterns: Iterable[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def print_paired_run_comparison(new_path: str, baseline_path: str) -> None:
    new_items = {str(item.get("id")): item for item in load_items(new_path)}
    baseline_items = {str(item.get("id")): item for item in load_items(baseline_path)}
    item_ids = sorted(set(new_items).intersection(baseline_items))
    transitions = Counter()
    analyzer_mutated = 0
    query_mutated = 0
    final_docs_mutated = 0
    prediction_mutated = 0
    for item_id in item_ids:
        new_item = new_items[item_id]
        baseline_item = baseline_items[item_id]
        new_output = output_of(new_item)
        baseline_output = output_of(baseline_item)
        new_trace = diagnostic_of(new_item)
        baseline_trace = diagnostic_of(baseline_item)
        analyzer_mutated += int(
            (new_trace.get("question_analysis") or {})
            != (baseline_trace.get("question_analysis") or {})
        )
        new_state = new_output.get("evidence_state") or {}
        baseline_state = baseline_output.get("evidence_state") or {}
        query_mutated += int(
            (new_state.get("query_history") or [])
            != (baseline_state.get("query_history") or [])
        )
        new_doc_ids = [str(doc.get("id")) for doc in new_output.get("retrieval_result") or []]
        baseline_doc_ids = [
            str(doc.get("id")) for doc in baseline_output.get("retrieval_result") or []
        ]
        final_docs_mutated += int(new_doc_ids != baseline_doc_ids)
        prediction_mutated += int(
            str(new_output.get("pred") or "") != str(baseline_output.get("pred") or "")
        )
        baseline_correct = float(metric_of(baseline_item).get("em", 0.0) or 0.0) > 0
        new_correct = float(metric_of(new_item).get("em", 0.0) or 0.0) > 0
        transitions[
            f"{'correct' if baseline_correct else 'wrong'}->{'correct' if new_correct else 'wrong'}"
        ] += 1

    count = len(item_ids)
    print("\npaired_run_comparison")
    print(f"baseline\t{baseline_path}")
    print(f"new\t{new_path}")
    print(f"paired_items\t{count}")
    for name, value in (
        ("question_analysis_mutation", analyzer_mutated),
        ("query_mutation", query_mutated),
        ("final_doc_mutation", final_docs_mutated),
        ("prediction_mutation", prediction_mutated),
    ):
        print(f"{name}_count\t{value}")
        print(f"{name}_rate\t{value / count:.4f}" if count else f"{name}_rate\t0.0000")
    print_counter("em_transitions", transitions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reviewer intermediate_data.json files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "examples/reviewer/result/*/intermediate_data.json",
            "examples/reviewer/result/*/*/*/intermediate_data.json",
            "examples/reviewer/result/*/*/*/*/intermediate_data.json",
        ],
        help="Files or glob patterns to analyze.",
    )
    parser.add_argument(
        "--compare-to",
        help="Baseline intermediate_data.json for paired item-level comparison.",
    )
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("No intermediate_data.json files found.")

    for path in paths:
        analyze_file(path)
    if args.compare_to:
        baseline_paths = expand_paths([args.compare_to])
        if len(paths) != 1 or len(baseline_paths) != 1:
            raise SystemExit("--compare-to requires exactly one new file and one baseline file.")
        print_paired_run_comparison(paths[0], baseline_paths[0])


if __name__ == "__main__":
    main()
