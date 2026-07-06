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

        policy = output_of(item).get("adoption_policy") or {}
        if policy.get("candidate_source"):
            candidate_sources[str(policy.get("candidate_source"))] += 1

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
        if answer_policy_mismatch(item):
            policy_mismatch += 1
            if f1 == 0:
                policy_mismatch_wrong += 1
        if bucket == "gold_added_by_adaptive" and f1 > 0:
            case_candidates["adaptive_added_gold_answered"] += 1
        if gold.get("final_gold_title_recall", 0.0) > 0 and f1 == 0:
            case_candidates["retrieved_gold_but_wrong"] += 1
        if coverage.get("coverage", 0.0) >= 0.99 and f1 == 0:
            case_candidates["high_coverage_but_wrong"] += 1
            high_coverage_wrong_by_answer_type[str(answer_type or "unknown")] += 1
        if forced and bucket == "gold_not_found":
            case_candidates["forced_gold_missing"] += 1

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
        total_answer_support = answer_supported_count + answer_unsupported_count
        print(f"answer_unsupported_rate\t{answer_unsupported_count / total_answer_support:.4f}" if total_answer_support else "answer_unsupported_rate\t0.0000")
        print(f"verifier_enabled_rate\t{verifier_enabled_count / traced:.4f}")
    print_counter("pipeline_versions", pipeline_versions)
    print_counter("state_schema_versions", state_schema_versions)
    print_counter("slot_statuses", slot_statuses)
    print_counter("answer_types", answer_types)
    print_counter("repair_methods", repair_methods)
    print_counter("candidate_sources", candidate_sources)
    print_counter("state_consistency_blocked_reasons", state_consistency_blocked_reasons)
    print_counter("high_coverage_wrong_by_answer_type", high_coverage_wrong_by_answer_type)
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
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("No intermediate_data.json files found.")

    for path in paths:
        analyze_file(path)


if __name__ == "__main__":
    main()
