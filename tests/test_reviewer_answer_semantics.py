import sys
import types
import unittest
from pathlib import Path


REVIEWER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEWER_DIR))

from reviewer_answer_semantics import (  # noqa: E402
    annotate_slots_with_intent,
    build_answer_intent,
    evaluate_answer_semantics,
    extract_comparison_options,
    infer_effective_question_type,
    repair_answer_slot_topology,
    resolve_claim_binding,
)
from reviewer_state import ReviewState  # noqa: E402


def comparison_state(question, claims):
    intent = build_answer_intent(question, "comparison", "phrase", [])
    slots = annotate_slots_with_intent(
        [
            {"slot_id": "slot_1", "slot_role": "comparison_operand", "status": "supported"},
            {"slot_id": "slot_2", "slot_role": "comparison_operand", "status": "supported"},
            {"slot_id": "slot_3", "slot_role": "final_answer", "status": "missing"},
        ],
        intent,
    )
    return {
        "question_type": "comparison",
        "answer_type": intent["expected_answer_type"],
        "answer_intent": intent,
        "answer_slots": slots,
        "claims": claims,
        "decision": "continue_retrieval",
    }


class AnswerIntentTests(unittest.TestCase):
    def test_was_or_comparison_returns_option_not_yes_no(self):
        intent = build_answer_intent(
            "Was Semsettin Bas or Gwenc'Hlan Le Scouezec born first?",
            "comparison",
            "yes_no",
            [],
        )
        self.assertEqual("compare_min", intent["operation"])
        self.assertEqual("person", intent["expected_answer_type"])
        self.assertEqual(["Semsettin Bas", "Gwenc'Hlan Le Scouezec"], intent["option_entities"])

    def test_comparison_executor_maps_value_back_to_option(self):
        question = "Who died first, Fleetwood Sheppard or George William Whitaker?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Fleetwood Sheppard",
                "predicate": "died", "object": "November 1698", "text": "Fleetwood Sheppard died in November 1698.",
                "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "George William Whitaker",
                "predicate": "died", "object": "November 1901",
                "text": "George William Whitaker died in November 1901.", "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(question, state, deterministic_commit_scope="strict_comparison")
        self.assertEqual("executed", trace["comparison_execution_status"])
        self.assertEqual("Fleetwood Sheppard", trace["answer_decision"]["selected_answer_candidate"])
        self.assertEqual("deterministic_commit", trace["answer_decision"]["commit_mode"])

    def test_comparison_requires_structured_operands(self):
        question = "Who died first, Fleetwood Sheppard or George William Whitaker?"
        state = comparison_state(question, [
            {"claim_id": "c1", "slot_id": "slot_1", "text": "Fleetwood Sheppard died in 1698.", "support_status": "support"},
            {"claim_id": "c2", "slot_id": "slot_2", "text": "George William Whitaker died in 1901.", "support_status": "support"},
        ])
        trace = evaluate_answer_semantics(question, state, deterministic_commit_scope="strict_comparison")
        self.assertEqual("incomplete_operands", trace["comparison_execution_status"])
        self.assertNotEqual("deterministic_commit", trace["answer_decision"]["commit_mode"])

    def test_numeric_comparison_returns_question_option(self):
        question = "Which mall, Wilton Mall or Viaport Rotterdam, has had more owners?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Wilton Mall",
                "predicate": "owner count", "object": "2", "text": "Wilton Mall had 2 owners.",
                "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "Viaport Rotterdam",
                "predicate": "owner count", "object": "3", "text": "Viaport Rotterdam had 3 owners.",
                "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(question, state, deterministic_commit_scope="strict_comparison")
        self.assertEqual("Viaport Rotterdam", trace["answer_decision"]["selected_answer_candidate"])

    def test_equality_comparison_returns_yes_no(self):
        question = "Were Scott Derrickson and Ed Wood of the same nationality?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Scott Derrickson",
                "predicate": "nationality", "object": "American", "text": "Scott Derrickson is American.",
                "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "Ed Wood",
                "predicate": "nationality", "object": "American", "text": "Ed Wood is American.",
                "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(question, state, deterministic_commit_scope="strict_comparison")
        self.assertEqual("yes", trace["answer_decision"]["selected_answer_candidate"])

    def test_default_mode_only_records_comparison_counterfactual(self):
        question = "Who died first, Fleetwood Sheppard or George William Whitaker?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Fleetwood Sheppard",
                "predicate": "died", "object": "1698", "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "George William Whitaker",
                "predicate": "died", "object": "1901", "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(question, state)
        self.assertEqual("Fleetwood Sheppard", trace["counterfactual_answer_candidate"])
        self.assertEqual("guided_generation", trace["answer_decision"]["commit_mode"])
        self.assertIsNone(trace["answer_decision"]["selected_answer_candidate"])

    def test_swapped_operand_slots_are_not_executable(self):
        question = "Who died first, Fleetwood Sheppard or George William Whitaker?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "George William Whitaker",
                "predicate": "died", "object": "1901", "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "Fleetwood Sheppard",
                "predicate": "died", "object": "1698", "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(
            question, state, deterministic_commit_scope="strict_comparison"
        )
        self.assertFalse(trace["strict_comparison_executable"])
        self.assertNotEqual("deterministic_commit", trace["answer_decision"]["commit_mode"])

    def test_comma_inside_option_title_is_preserved(self):
        question = "Which film has the director who died later, Good Morning, Little Countess or Vogue La Galère?"
        self.assertEqual(
            ["Good Morning, Little Countess", "Vogue La Galère"],
            extract_comparison_options(question),
        )

    def test_recent_release_direct_comparison_is_strictly_executable(self):
        question = "Which film was released more recently, Lo Zappatore or Placer Sangriento?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Lo Zappatore",
                "predicate": "release_date", "object": "5 December 1980", "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "Placer Sangriento",
                "predicate": "release_date", "object": "19 January 2018", "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(
            question, state, deterministic_commit_scope="strict_comparison"
        )
        self.assertTrue(trace["strict_comparison_executable"])
        self.assertEqual("Placer Sangriento", trace["answer_decision"]["selected_answer_candidate"])

    def test_nested_director_birth_comparison_requires_full_lineage(self):
        question = "Which film has the director born first, Once A Gentleman or The Girl In White?"
        intent = build_answer_intent(question, "comparison", "title_or_entity", [])
        slots = annotate_slots_with_intent([
            {"slot_id": "film_1", "slot_role": "intermediate_entity", "status": "supported"},
            {"slot_id": "date_1", "slot_role": "comparison_operand", "depends_on": ["film_1"], "status": "supported"},
            {"slot_id": "film_2", "slot_role": "intermediate_entity", "status": "supported"},
            {"slot_id": "date_2", "slot_role": "comparison_operand", "depends_on": ["film_2"], "status": "supported"},
        ], intent)
        slots[1]["comparison_option"] = "Once A Gentleman"
        slots[3]["comparison_option"] = "The Girl In White"
        state = {
            "question_type": "comparison", "answer_type": "title_or_entity",
            "answer_intent": intent, "answer_slots": slots, "decision": "continue_retrieval",
            "claims": [
                {"claim_id": "b1", "slot_id": "film_1", "subject": "Once A Gentleman", "predicate": "director", "object": "Director One", "support_status": "support"},
                {"claim_id": "v1", "slot_id": "date_1", "subject": "Director One", "predicate": "born", "object": "1887", "support_status": "support"},
                {"claim_id": "b2", "slot_id": "film_2", "subject": "The Girl In White", "predicate": "director", "object": "Director Two", "support_status": "support"},
                {"claim_id": "v2", "slot_id": "date_2", "subject": "Director Two", "predicate": "born", "object": "1938", "support_status": "support"},
            ],
        }
        trace = evaluate_answer_semantics(
            question, state, deterministic_commit_scope="strict_comparison"
        )
        self.assertTrue(trace["strict_comparison_executable"])
        self.assertEqual("Once A Gentleman", trace["answer_decision"]["selected_answer_candidate"])

    def test_mixed_date_precision_is_not_strictly_executable(self):
        question = "Which film was released more recently, Film A or Film B?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Film A",
                "predicate": "release_date", "object": "1980", "support_status": "support",
            },
            {
                "claim_id": "c2", "slot_id": "slot_2", "subject": "Film B",
                "predicate": "release_date", "object": "5 December 1980", "support_status": "support",
            },
        ])
        trace = evaluate_answer_semantics(
            question, state, deterministic_commit_scope="strict_comparison"
        )
        self.assertFalse(trace["strict_comparison_executable"])
        self.assertIn(
            "operand_date_precision_mismatch",
            trace["comparison_execution"]["rejection_reasons"],
        )

    def test_strict_provenance_does_not_rebind_to_another_document(self):
        question = "Who died first, Fleetwood Sheppard or George William Whitaker?"
        state = comparison_state(question, [
            {
                "claim_id": "c1", "doc_id": "wrong", "slot_id": "slot_1",
                "subject": "Fleetwood Sheppard", "predicate": "died", "object": "1698",
                "support_status": "support",
            },
            {
                "claim_id": "c2", "doc_id": "george", "slot_id": "slot_2",
                "subject": "George William Whitaker", "predicate": "died", "object": "1901",
                "support_status": "support",
            },
        ])
        docs = [
            {"id": "wrong", "title": "Unrelated", "contents": "Unrelated material from 1698."},
            {"id": "fleetwood", "title": "Fleetwood Sheppard", "contents": "Fleetwood Sheppard died in 1698."},
            {"id": "george", "title": "George William Whitaker", "contents": "George William Whitaker died in 1901."},
        ]
        trace = evaluate_answer_semantics(
            question, state, docs=docs, deterministic_commit_scope="strict_comparison"
        )
        self.assertFalse(trace["strict_comparison_executable"])
        fleetwood = trace["comparison_lineage"][0]
        self.assertFalse(fleetwood["provenance_valid"])
        self.assertEqual(["fleetwood"], fleetwood["alternative_value_provenance_doc_ids"])


class IntentRegressionTests(unittest.TestCase):
    def test_inverse_claim_returns_subject(self):
        binding = resolve_claim_binding(
            {
                "subject": "Ted Turner", "predicate": "founded",
                "object": "CNN", "text": "Ted Turner founded CNN",
            },
            [],
            "Who founded CNN?",
        )
        self.assertTrue(binding["entity_binding_valid"])
        self.assertEqual("Ted Turner", binding["candidate"])
        self.assertEqual("object_anchor_to_subject", binding["binding_direction"])

    def test_forward_claim_returns_object(self):
        binding = resolve_claim_binding(
            {
                "subject": "Ada Lovelace", "predicate": "born in",
                "object": "London", "text": "Ada Lovelace was born in London",
            },
            [],
            "Where was Ada Lovelace born?",
        )
        self.assertTrue(binding["entity_binding_valid"])
        self.assertEqual("London", binding["candidate"])

    def test_intermediate_final_topology_repairs_single_hop_to_bridge(self):
        slots = [
            {"slot_id": "slot_1", "slot_role": "intermediate_entity", "status": "supported"},
            {"slot_id": "slot_2", "slot_role": "final_answer", "status": "supported"},
        ]
        repaired = repair_answer_slot_topology(
            slots, "Who was the wife of the founder of CNN?", "single_hop"
        )
        effective, reason = infer_effective_question_type(
            "single_hop", repaired, "Who was the wife of the founder of CNN?"
        )
        self.assertEqual("bridge", effective)
        self.assertEqual("intermediate_final_slot_topology", reason)
        self.assertEqual("final_answer", repaired[1]["slot_role"])

    def test_wrong_spouse_chain_has_no_validated_candidate(self):
        question = "Who was the second wife of the founder of CNN?"
        state = {
            "question_type": "bridge", "answer_type": "person",
            "answer_slots": [
                {"slot_id": "slot_1", "slot_role": "intermediate_entity", "status": "supported"},
                {"slot_id": "slot_2", "slot_role": "final_answer", "depends_on": ["slot_1"], "status": "supported", "target_relation": "wife"},
            ],
            "claims": [
                {"claim_id": "c1", "slot_id": "slot_1", "subject": "Ted Turner", "predicate": "founded", "object": "CNN", "support_status": "support"},
                {"claim_id": "c2", "slot_id": "slot_2", "subject": "Hadassah Lieberman", "predicate": "second wife of", "object": "Joe Lieberman", "support_status": "support"},
            ],
            "decision": "ready_to_answer",
        }
        trace = evaluate_answer_semantics(question, state)
        self.assertEqual("Ted Turner", trace["resolved_intermediate_entities"][0]["entity"])
        self.assertFalse(trace["relation_grounded"])
        self.assertIsNone(trace["counterfactual_answer_candidate"])

    def test_capacity_slot_beats_distractor_role_intent(self):
        intent = build_answer_intent(
            "The arena where the Lewiston Maineiacs played can seat how many people?",
            "bridge", "number",
            [{"slot_role": "final_answer", "target_relation": "capacity"}],
            {"target_relation": "role"},
        )
        self.assertEqual("capacity", intent["relation_family"])

    def test_seated_capacity_rejects_generic_total_capacity(self):
        question = "The arena where the Lewiston Maineiacs played can seat how many people?"
        intent = build_answer_intent(
            question, "single_hop", "number",
            [{"slot_id": "slot_1", "slot_role": "direct_answer", "target_relation": "capacity"}],
        )
        state = {
            "question_type": "single_hop", "answer_type": "number", "answer_intent": intent,
            "answer_slots": [{
                "slot_id": "slot_1", "slot_role": "direct_answer", "status": "supported",
                "target_relation": "capacity",
            }],
            "claims": [{
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Androscoggin Bank Colisée",
                "predicate": "capacity", "object": "4,000",
                "text": "Androscoggin Bank Colisée has a capacity of 4,000",
                "support_status": "support",
            }],
            "decision": "ready_to_answer",
        }
        trace = evaluate_answer_semantics(question, state)
        self.assertFalse(trace["relation_grounded"])
        self.assertIsNone(trace["counterfactual_answer_candidate"])

    def test_stage_name_does_not_inherit_birth_relation(self):
        intent = build_answer_intent(
            "Robert C Seger was born May 6 1945. What was the rock singer's stage name?",
            "bridge", "person",
            [{"slot_role": "final_answer", "target_relation": "stage_name"}],
            {"target_relation": "birth_date"},
        )
        self.assertEqual("generic", intent["relation_family"])

    def test_most_recently_is_compare_max(self):
        intent = build_answer_intent(
            "Which board game was published most recently, Pirate's Cove or Catan?",
            "comparison", "title_or_entity", [],
        )
        self.assertEqual("compare_max", intent["operation"])

    def test_newer_is_compare_max(self):
        intent = build_answer_intent(
            "Which film is newer, The Apple Dumpling Gang or Heavyweights?",
            "comparison", "title_or_entity", [],
        )
        self.assertEqual("compare_max", intent["operation"])

    def test_abolished_relation_does_not_become_nationality(self):
        intent = build_answer_intent(
            "When was the country Merewalh was a citizen of abolished?",
            "bridge", "date",
            [{"slot_role": "final_answer", "target_relation": "abolished"}],
            {"target_relation": "nationality"},
        )
        self.assertEqual("end_date", intent["relation_family"])

    def test_slot_topology_repair_does_not_promote_status(self):
        repaired = repair_answer_slot_topology(
            [{"slot_id": "slot_1", "slot_role": "final_answer", "status": "missing"}],
            "Who died first, A Person or B Person?", "comparison",
        )
        self.assertEqual(2, sum(slot["slot_role"] == "comparison_operand" for slot in repaired))
        self.assertTrue(all(slot.get("status") != "supported" for slot in repaired))


class RelationGroundingTests(unittest.TestCase):
    def bridge_state(self, predicate="studied at", object_value="Slade School of Art"):
        question = "Where did the painter of The Mud Bath study?"
        intent = build_answer_intent(question, "bridge", "location", [])
        slots = annotate_slots_with_intent([
            {"slot_id": "slot_1", "slot_role": "intermediate_entity", "status": "supported"},
            {"slot_id": "slot_2", "slot_role": "final_answer", "status": "supported"},
        ], intent)
        state = {
            "question_type": "bridge",
            "answer_type": "location",
            "answer_intent": intent,
            "answer_slots": slots,
            "claims": [
                {
                    "claim_id": "c1", "slot_id": "slot_1", "predicate": "painter",
                    "object": "David Bomberg", "text": "The painter was David Bomberg.",
                    "support_status": "support",
                },
                {
                    "claim_id": "c2", "slot_id": "slot_2", "subject": "David Bomberg",
                    "predicate": predicate, "object": object_value,
                    "text": f"David Bomberg {predicate} {object_value}.", "support_status": "support",
                },
            ],
            "decision": "ready_to_answer",
        }
        return question, state

    def test_bridge_requires_subject_relation_and_object(self):
        question, state = self.bridge_state()
        trace = evaluate_answer_semantics(question, state)
        self.assertTrue(trace["relation_grounded"])
        self.assertEqual("Slade School of Art", trace["counterfactual_answer_candidate"])
        self.assertNotEqual("deterministic_commit", trace["answer_decision"]["commit_mode"])

    def test_wrong_relation_is_not_committed(self):
        question, state = self.bridge_state("was born in", "Birmingham")
        trace = evaluate_answer_semantics(question, state)
        self.assertFalse(trace["relation_grounded"])
        self.assertTrue(trace["grounded_but_wrong_relation"])
        self.assertNotEqual("deterministic_commit", trace["answer_decision"]["commit_mode"])

    def test_conflicting_valid_candidates_fall_back_to_generation(self):
        question, state = self.bridge_state()
        state["claims"].append({
            "claim_id": "c3", "slot_id": "slot_2", "subject": "David Bomberg",
            "predicate": "studied at", "object": "Birmingham School of Art",
            "text": "David Bomberg studied at Birmingham School of Art.", "support_status": "support",
        })
        trace = evaluate_answer_semantics(question, state)
        self.assertEqual("guided_generation", trace["answer_decision"]["commit_mode"])
        self.assertEqual("no_unique_validated_candidate", trace["answer_decision"]["commit_reason"])


class StateCompatibilityTests(unittest.TestCase):
    def test_legacy_unknown_fields_survive_v2_7_normalization(self):
        state = ReviewState.from_dict({
            "pipeline_version": "ver2_6",
            "state_schema_version": "reviewer_state_v2_6",
            "question_type": "bridge",
            "answer_slots": [{
                "slot_id": "slot_1", "description": "legacy slot", "slot_role": "final_answer",
                "status": "supported", "legacy_slot_field": "keep",
            }],
            "claims": [{
                "claim_id": "c1", "doc_id": "d1", "slot_id": "slot_1", "text": "Legacy claim.",
                "support_status": "support", "legacy_claim_field": "keep",
            }],
            "legacy_state_field": "keep",
        }).to_dict()
        self.assertEqual("keep", state["legacy_state_field"])
        self.assertEqual("keep", state["answer_slots"][0]["legacy_slot_field"])
        self.assertEqual("keep", state["claims"][0]["legacy_claim_field"])


class PipelineCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flashrag = types.ModuleType("flashrag")
        pipeline_module = types.ModuleType("flashrag.pipeline")
        utils_module = types.ModuleType("flashrag.utils")
        prompt_module = types.ModuleType("flashrag.prompt")
        pipeline_module.BasicPipeline = type("BasicPipeline", (), {})
        utils_module.get_retriever = lambda config: None
        utils_module.get_generator = lambda config: None
        prompt_module.PromptTemplate = type("PromptTemplate", (), {})
        sys.modules.update({
            "flashrag": flashrag,
            "flashrag.pipeline": pipeline_module,
            "flashrag.utils": utils_module,
            "flashrag.prompt": prompt_module,
        })
        from reviewer_pipeline import ReviewerPipeline
        cls.pipeline_class = ReviewerPipeline

    def test_batch_helper_skips_committed_items_and_preserves_order(self):
        class CountingGenerator:
            def __init__(self):
                self.calls = []

            def generate(self, prompts):
                self.calls.append(list(prompts))
                return [f"generated-{prompt}" for prompt in prompts]

        pipeline = self.pipeline_class.__new__(self.pipeline_class)
        pipeline.enable_deterministic_answer_commit = True
        pipeline.batch_size = 16
        pipeline.generator = CountingGenerator()
        pipeline.config = {"dataset_name": "hotpotqa"}
        committed = lambda answer: {
            "answer_decision": {
                "commit_mode": "deterministic_commit",
                "selected_answer_candidate": answer,
            },
            "selected_answer_candidate": answer,
        }
        policies = [
            committed("Fleetwood Sheppard"),
            {"answer_decision": {"commit_mode": "guided_generation"}},
            committed("yes"),
        ]
        states = [{"question_type": "bridge"} for _ in policies]
        answers = pipeline._generate_final_answers(["p1", "p2", "p3"], policies, states)
        self.assertEqual(["Fleetwood Sheppard", "generated-p2", "yes"], answers)
        self.assertEqual([["p2"]], pipeline.generator.calls)

    def test_commit_disabled_sends_all_items_to_generator(self):
        class CountingGenerator:
            def __init__(self):
                self.calls = []

            def generate(self, prompts):
                self.calls.append(list(prompts))
                return [f"generated-{prompt}" for prompt in prompts]

        pipeline = self.pipeline_class.__new__(self.pipeline_class)
        pipeline.enable_deterministic_answer_commit = False
        pipeline.batch_size = 16
        pipeline.generator = CountingGenerator()
        pipeline.config = {"dataset_name": "hotpotqa"}
        policy = {
            "answer_decision": {
                "commit_mode": "deterministic_commit",
                "selected_answer_candidate": "counterfactual",
            }
        }
        answers = pipeline._generate_final_answers(["p1"], [policy], [{}])
        self.assertEqual(["generated-p1"], answers)
        self.assertEqual([["p1"]], pipeline.generator.calls)

    def test_diagnostic_grounding_does_not_change_control_state(self):
        pipeline = self.pipeline_class.__new__(self.pipeline_class)
        pipeline.enable_relation_grounding = True
        pipeline.relation_grounding_mode = "diagnostic"
        pipeline._build_relation_grounding_trace = lambda state, docs, question: {
            "relation_grounded": False,
            "chain_grounding_trace": {"chain_grounded": False},
            "answer_decision": {"commit_mode": "guided_generation"},
            "answer_intent": {},
        }
        state = {
            "decision": "ready_to_answer",
            "failure_reason": None,
            "retrieval_plan": [{"slot_id": "slot_1", "query": "original"}],
        }
        before = (state["decision"], state["failure_reason"], list(state["retrieval_plan"]))
        changed = pipeline._apply_chain_grounding_ready_guard(state, [], "question")
        after = (state["decision"], state["failure_reason"], state["retrieval_plan"])
        self.assertFalse(changed)
        self.assertEqual(before, after)

    def test_diagnostic_mode_does_not_leak_preferred_candidate(self):
        state = {
            "question_type": "single_hop", "answer_type": "person",
            "answer_slots": [{
                "slot_id": "slot_1", "slot_role": "direct_answer", "status": "supported",
                "target_relation": "founded",
            }],
            "claims": [{
                "claim_id": "c1", "slot_id": "slot_1", "subject": "Ted Turner",
                "predicate": "founded", "object": "CNN", "support_status": "support",
            }],
            "decision": "ready_to_answer", "answer_candidate": None,
        }

        def make_pipeline(enabled, mode):
            pipeline = self.pipeline_class.__new__(self.pipeline_class)
            pipeline.enable_relation_grounding = enabled
            pipeline.relation_grounding_mode = mode
            pipeline.enable_comparison_executor = True
            pipeline.enable_deterministic_answer_commit = False
            pipeline.deterministic_commit_scope = "none"
            pipeline.config = {"dataset_name": "bamboogle"}
            return pipeline

        diagnostic = make_pipeline(True, "diagnostic")
        disabled = make_pipeline(False, "diagnostic")
        control = make_pipeline(True, "control")
        diagnostic_policy = diagnostic._build_adoption_policy(
            dict(state), [], question="Who founded CNN?"
        )
        disabled_policy = disabled._build_adoption_policy(
            dict(state), [], question="Who founded CNN?"
        )
        control_policy = control._build_adoption_policy(
            dict(state), [], question="Who founded CNN?"
        )
        self.assertEqual([], diagnostic_policy["answer_candidates"])
        self.assertFalse(diagnostic_policy["diagnostic_candidate_leakage"])
        self.assertIsNone(diagnostic._fallback_answer_from_policy(diagnostic_policy))
        self.assertEqual(
            disabled._render_adoption_policy(disabled_policy),
            diagnostic._render_adoption_policy(diagnostic_policy),
        )
        self.assertEqual(["Ted Turner"], control_policy["answer_candidates"])
        self.assertEqual("Ted Turner", control_policy["selected_answer_candidate"])
        self.assertEqual("Ted Turner", control._fallback_answer_from_policy(control_policy))

    def test_question_analysis_repairs_bridge_type_from_slots(self):
        pipeline = self.pipeline_class.__new__(self.pipeline_class)
        parsed = {
            "question_type": "single_hop",
            "answer_slots": [
                {
                    "slot_id": "slot_1", "slot_role": "intermediate_entity",
                    "description": "Identify the founder of CNN", "status": "missing",
                },
                {
                    "slot_id": "slot_2", "slot_role": "final_answer",
                    "description": "Find the founder's second wife", "status": "missing",
                },
            ],
        }
        state = pipeline._normalize_question_analysis(
            parsed, "Who was the second wife of the founder of CNN?"
        )
        self.assertEqual("bridge", state["question_type"])
        self.assertEqual("single_hop", state["original_question_type"])
        self.assertTrue(state["question_type_repaired"])
        self.assertEqual("final_answer", state["answer_slots"][1]["slot_role"])


if __name__ == "__main__":
    unittest.main()
