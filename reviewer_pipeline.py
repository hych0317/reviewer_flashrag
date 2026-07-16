"""
Reviewer Pipeline: A custom RAG pipeline with LLM-based document review loop.

This pipeline adds a review phase before generation, where an LLM (reviewer) evaluates
retrieved documents for quality, sufficiency, and factual consistency. Based on the
evaluation, the reviewer either:
- Triggers re-retrieval with an optimized query and filters harmful documents
- Generates a meta-cognitive adoption strategy to guide the final generator

Author: Custom Implementation
Reference: FlashRAG Loop-based Pipelines (IterativePipeline, IRCOTPipeline, etc.)
"""

import json
import hashlib
import os
import re
import subprocess
from typing import List, Dict, Any, Tuple, Optional

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate

try:
    from reviewer_state import (
        PIPELINE_VERSION,
        PROMPT_VERSION,
        STATE_SCHEMA_VERSION,
        ReviewState,
        normalize_review_state,
        repair_slot_claim_consistency,
    )
except ImportError:
    from examples.reviewer.reviewer_state import (
        PIPELINE_VERSION,
        PROMPT_VERSION,
        STATE_SCHEMA_VERSION,
        ReviewState,
        normalize_review_state,
        repair_slot_claim_consistency,
    )

try:
    from reviewer_answer_semantics import (
        annotate_slots_with_intent,
        build_answer_intent,
        build_comparison_output_contract,
        evaluate_answer_semantics,
        infer_effective_question_type,
        repair_answer_slot_topology,
        validate_final_comparison_commit,
    )
except ImportError:
    from examples.reviewer.reviewer_answer_semantics import (
        annotate_slots_with_intent,
        build_answer_intent,
        build_comparison_output_contract,
        evaluate_answer_semantics,
        infer_effective_question_type,
        repair_answer_slot_topology,
        validate_final_comparison_commit,
    )


# ============================================================================
# Reviewer Prompt Templates
# ============================================================================

QUESTION_ANALYZER_SYSTEM_PROMPT = """Task: Analyze the question before retrieval review.

Classify the question and decompose it into the minimum required answer slots.

Question types:
- single_hop: one fact is enough
- bridge: one fact identifies an intermediate entity, another fact answers through it
- comparison: two or more entities must be compared
- ambiguous: the question may have multiple interpretations

Slot roles:
- intermediate_entity: an entity or relation needed before the final answer can be retrieved
- final_answer: the exact attribute, entity, yes/no result, or short phrase requested by the question
- comparison_operand: an entity attribute needed for a comparison
- direct_answer: a directly answerable fact for a single-hop question

For bridge questions, always create at least one intermediate_entity slot and one final_answer slot.
For comparison questions, create comparison_operand slots and one final_answer slot.

Output only valid JSON:
```json
{{
    "question_type": "single_hop | bridge | comparison | ambiguous",
    "answer_intent": {{
        "operation": "direct_lookup | bridge_lookup | compare_min | compare_max | compare_equal | unknown",
        "target_relation": "short relation requested by the question",
        "relation_family": "normalized relation family",
        "answer_role": "object | option | boolean",
        "expected_answer_type": "yes_no | person | date | location | title_or_entity | number | phrase",
        "option_entities": []
    }},
    "answer_slots": [
        {{
            "slot_id": "slot_1",
            "description": "specific fact needed to answer",
            "slot_role": "intermediate_entity | final_answer | comparison_operand | direct_answer",
            "depends_on": [],
            "target_relation": "relation required by this slot",
            "relation_family": "normalized relation family",
            "expected_answer_type": "expected object type",
            "comparison_option": null,
            "status": "missing",
            "supporting_claim_ids": [],
            "conflicting_claim_ids": []
        }}
    ],
    "explanation": "brief reason for the decomposition"
}}
```"""

QUESTION_ANALYZER_USER_PROMPT = """Question: {question}

Return the JSON analysis."""


EVIDENCE_REVIEWER_SYSTEM_PROMPT = """Task: Update an EvidenceState for a RAG system.

You receive a question, the current EvidenceState, and retrieved documents. Your job is to:
1. Extract only atomic claims that are relevant to the answer slots.
2. Attach each claim to a slot and document id.
3. Mark each slot as missing, partial, supported, or conflicted.
4. Decide whether to continue retrieval, answer now, or force an answer with insufficient evidence.
5. If retrieval is needed, produce slot-specific queries for only missing or conflicted slots.
6. Identify whether the final answer itself is directly supported, not merely an intermediate fact.

Rules:
- Do not invent claims that are not in the documents.
- Prefer short claims, one fact per claim.
- Use support_status values: support, contradict, irrelevant.
- Use decision values: continue_retrieval, ready_to_answer, force_answer.
- Use direct_answer_supported=true only when a short answer candidate directly answers the original question.
- If the final answer is not directly supported, decision must be continue_retrieval even when intermediate slots are supported.
- For bridge questions, use resolved intermediate entities in second-hop queries for the final_answer slot.
- If evidence is insufficient and more retrieval may help, decision must be continue_retrieval.
- If evidence remains insufficient but no useful query can be generated, decision must be force_answer.
- Output only valid JSON.

Output format:
```json
{{
    "slot_updates": [
        {{
            "slot_id": "slot_1",
            "status": "missing | partial | supported | conflicted",
            "supporting_claim_ids": [],
            "conflicting_claim_ids": []
        }}
    ],
    "claims": [
        {{
            "claim_id": "c1",
            "doc_id": "document id from the prompt",
            "slot_id": "slot_1",
            "text": "atomic factual claim",
            "subject": "claim subject or null",
            "predicate": "claim relation or null",
            "object": "claim object/value or null",
            "object_type": "person | date | location | title_or_entity | number | phrase | null",
            "support_status": "support | contradict | irrelevant",
            "round_id": 0
        }}
    ],
    "retrieval_plan": [
        {{
            "slot_id": "slot_1",
            "query": "precise search query for the missing or conflicted slot",
            "reason": "what evidence is missing"
        }}
    ],
    "docs_to_remove": [],
    "ignored_docs": [],
    "direct_answer_supported": false,
    "answer_type": "yes_no | person | date | location | title_or_entity | number | phrase",
    "answer_candidate": null,
    "answer_candidate_claim_ids": [],
    "decision": "continue_retrieval | ready_to_answer | force_answer",
    "failure_reason": null,
    "explanation": "brief evidence review"
}}
```"""

EVIDENCE_REVIEWER_USER_PROMPT = """Question: {question}

Current EvidenceState:
```json
{evidence_state}
```

Retrieved Documents:
{formatted_docs}

Current round id: {round_id}

Return the updated EvidenceState JSON."""


# Legacy prompt names are kept for demo_reviewer.py and older notebooks. The
# main ReviewerPipeline uses the QuestionAnalyzer/EvidenceReviewer templates
# above.
REVIEWER_SYSTEM_PROMPT = """Task: Evaluate retrieved documents for sufficiency and reliability.

Evaluation dimensions:
1. Information Sufficiency: Whether documents contain enough information to answer the question completely
2. Factual Consistency: Whether contradictions or conflicts exist between documents
3. Document Credibility: Whether documents are reliable; presence of harmful, misleading, or incorrect information

Output one of two decisions:
- NEED_MORE_INFO: documents are insufficient, conflicting, or unreliable
- READY_TO_ANSWER: documents are sufficient and reliable

Output only valid JSON:
```json
{{
    "decision": "NEED_MORE_INFO or READY_TO_ANSWER",
    "evaluation": {{
        "sufficiency": "sufficient/partial/insufficient",
        "conflicts": "none/minor/major",
        "credibility": "high/medium/low",
        "problematic_docs": []
    }},
    "explanation": "Brief explanation",
    "new_query": "New search query or null",
    "docs_to_remove": [],
    "adoption_strategy": "Strategy or null"
}}
```"""

REVIEWER_USER_PROMPT = """Question: {question}

Retrieved Documents:
{formatted_docs}

Output evaluation in the specified JSON format."""


FORCED_ADOPTION_STRATEGY_PROMPT = """Maximum review iterations reached. Generate final adoption policy.

Question: {question}

Current Documents:
{formatted_docs}

EvidenceState:
```json
{evidence_state}
```

Review History:
{review_summary}

Policy rules:

**For closed-book QA benchmarks that require an answer**:
- When evidence is incomplete or max rounds are reached, still produce the best supported short answer from the provided documents.
- Do not answer "unknown", "insufficient evidence", or "cannot determine" unless the documents contain no plausible answer candidate.
- Do not introduce facts that are absent from the documents.

**Documents COMPLETELY IRRELEVANT or HARMFUL**:
- ignored_docs: all document indices
- insufficient_slots: all unresolved slot ids

**Documents have LIMITED RELEVANCE**:
- trusted_claims: only directly supported claims
- insufficient_slots: unresolved slot ids

**Documents are SUFFICIENT**:
- trusted_claims: all answer-supporting claims
- must_use_slots: required slot ids

Output format (JSON):
```json
{{
    "trusted_claims": [],
    "answer_candidates": [],
    "ignored_docs": [],
    "conflict_resolution": [],
    "must_use_slots": [],
    "insufficient_slots": [],
    "answer_style": "short_answer",
    "failure_reason": "max_rounds"
}}
```"""

FINAL_GENERATION_SYSTEM_PROMPT = """Answer questions strictly based on the provided documents and evidence guidance below.

Evidence Guidance:
{adoption_strategy}

Use the supported facts first. Ignore documents marked as unreliable. If a required fact is incomplete, answer with the best supported short answer only when possible.
For benchmark QA, still provide the most likely short answer from the documents when evidence is partial; avoid uncertainty-only answers unless no plausible answer is present.

Only output the final answer. Do not explain your reasoning. Do not output JSON, field names, or internal policy terms such as trusted_claims, insufficient_slots, must_use_slots, ignored_docs, or conflict_resolution.
Prefer a concise entity, date, number, location, title, yes/no answer, or short phrase.
Expected output format examples:
- Question: who got the first nobel prize in physics  
Answer: Wilhelm Conrad Röntgen
- Question: what does hp mean in war and order  
Answer: hit points or health points

The following are given documents."""

FINAL_GENERATION_USER_PROMPT = """Documents:
{reference}

Question: {question}

Answer:"""


class ReviewerPipeline(BasicPipeline):
    """
    A custom RAG pipeline with an LLM-based document review loop.

    The pipeline flow:
    1. Initial retrieval: Q -> Retriever -> DOCS
    2. Review loop (max_iterations):
       a. Reviewer LLM evaluates Q + DOCS
       b. If NEED_MORE_INFO:
          - Generate new query Q_new
          - Retrieve new documents
          - Merge documents and remove harmful/incorrect ones
          - Continue loop
       c. If READY_TO_ANSWER:
          - Generate adoption strategy E
          - Exit loop
    3. Final generation: Q + cleaned DOCS + strategy E -> Generator -> Answer

    Args:
        config: FlashRAG configuration
        prompt_template: Optional custom prompt template for final generation
        max_iterations: Maximum number of review iterations (default: 3)
        retriever: Optional pre-initialized retriever
        generator: Optional pre-initialized generator (used for final answer generation)
        reviewer: Optional pre-initialized LLM for review (if None, uses generator)
        max_docs: Maximum number of documents to keep after merging (default: 10)
    """

    def __init__(
        self,
        config,
        prompt_template: Optional[PromptTemplate] = None,
        max_iterations: int = 3,
        retriever=None,
        generator=None,
        reviewer=None,
        max_docs: int = 10,
        batch_size: int = 256,
        serper_retriever=None,
    ):
        super().__init__(config, prompt_template)

        # Initialize components
        self.retriever = retriever if retriever is not None else get_retriever(config)
        self.generator = generator if generator is not None else get_generator(config)

        # Reviewer can be a separate LLM or share the generator
        # If separate reviewer is needed, pass it via reviewer parameter
        self.reviewer = reviewer if reviewer is not None else self.generator

        self.max_iterations = max_iterations
        self.max_docs = max_docs
        self.max_slot_queries = self.config["reviewer_max_slot_queries"] or 3
        self.final_max_docs = min(self.max_docs, self.config["reviewer_final_max_docs"] or 8)
        self.batch_size = batch_size  # Max prompts per generate call
        self.pipeline_version = self.config["pipeline_version"] or PIPELINE_VERSION
        self.state_schema_version = self.config["state_schema_version"] or STATE_SCHEMA_VERSION
        self.prompt_version = self.config["prompt_version"] or PROMPT_VERSION
        self.enable_state_machine_executor = self.config["enable_state_machine_executor"]
        if self.enable_state_machine_executor is None:
            self.enable_state_machine_executor = True
        self.enable_structured_retrieval_plan = bool(self.config["enable_structured_retrieval_plan"])
        self.enable_answer_verifier = bool(self.config["enable_answer_verifier"])
        self.enable_hybrid_retrieval = bool(self.config["enable_hybrid_retrieval"])
        relation_flag = self._config_value("enable_relation_grounding", True)
        comparison_flag = self._config_value("enable_comparison_executor", True)
        comparison_branch_mode = str(
            self._config_value("comparison_branch_mode", "baseline") or "baseline"
        )
        comparison_mode_aliases = {
            "diagnostic": "baseline",
            "final_only": "strict_commit",
        }
        comparison_finalization_mode = comparison_mode_aliases.get(
            comparison_branch_mode, comparison_branch_mode
        )
        if comparison_finalization_mode not in {"baseline", "strict_commit", "control"}:
            comparison_finalization_mode = "baseline"
        relation_mode = str(self._config_value("relation_grounding_mode", "diagnostic") or "diagnostic")
        if relation_mode not in {"diagnostic", "control"}:
            relation_mode = "diagnostic"
        commit_scope = str(self._config_value("deterministic_commit_scope", "none") or "none")
        if commit_scope not in {"none", "strict_comparison"}:
            commit_scope = "none"
        commit_flag = self._config_value("enable_deterministic_answer_commit", False)
        self.enable_relation_grounding = bool(relation_flag)
        self.relation_grounding_mode = relation_mode
        self.enable_comparison_executor = bool(comparison_flag)
        self.comparison_branch_mode = comparison_branch_mode
        self.comparison_finalization_mode = comparison_finalization_mode
        self.deterministic_commit_scope = commit_scope
        self.enable_deterministic_answer_commit = bool(commit_flag) and commit_scope != "none"

        # Serper fallback configuration
        self.use_serper_fallback = config["use_serper_fallback"] or False
        self.serper_retriever = serper_retriever
        self._serper_initialized = serper_retriever is not None

        serper_settings = config["serper_fallback_settings"] or {}
        self.serper_problematic_ratio_threshold = serper_settings.get("problematic_ratio_threshold", 0.5)
        self.serper_topk = serper_settings.get("serper_topk", 5)
        self.serper_merge_strategy = serper_settings.get("merge_strategy", "prepend")

        # Create question analyzer and evidence reviewer prompt templates
        self.question_analyzer_prompt_template = PromptTemplate(
            config=config,
            system_prompt=QUESTION_ANALYZER_SYSTEM_PROMPT,
            user_prompt=QUESTION_ANALYZER_USER_PROMPT,
        )
        self.evidence_reviewer_prompt_template = PromptTemplate(
            config=config,
            system_prompt=EVIDENCE_REVIEWER_SYSTEM_PROMPT,
            user_prompt=EVIDENCE_REVIEWER_USER_PROMPT,
        )

        # Create forced adoption strategy prompt template (for max iterations case)
        self.forced_adoption_prompt_template = PromptTemplate(
            config=config,
            system_prompt="You are a document reviewer that must now provide a final adoption strategy.",
            user_prompt=FORCED_ADOPTION_STRATEGY_PROMPT,
        )

        # Override final generation prompt if not provided
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                config=config,
                system_prompt=FINAL_GENERATION_SYSTEM_PROMPT,
                user_prompt=FINAL_GENERATION_USER_PROMPT,
            )

    def _init_serper_retriever(self) -> bool:
        """
        Lazily initialize the SerperRetriever.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._serper_initialized:
            return self.serper_retriever is not None

        self._serper_initialized = True

        if not self.use_serper_fallback:
            return False

        serper_api_key = self.config["serper_api_key"]
        if not serper_api_key:
            print("[Warning] Serper fallback enabled but serper_api_key not configured. Disabling fallback.")
            self.use_serper_fallback = False
            return False

        try:
            from flashrag.retriever import SerperRetriever

            serper_config = {
                "retrieval_topk": self.serper_topk,
                "serper_api_key": serper_api_key,
                "serper_search_type": self.config["serper_search_type"] or "search",
                "serper_location": self.config["serper_location"],
                "serper_gl": self.config["serper_gl"],
                "serper_hl": self.config["serper_hl"] or "en",
                "index_path": None,
                "corpus_path": None,
                "save_retrieval_cache": False,
                "use_retrieval_cache": False,
                "use_reranker": False,
            }

            self.serper_retriever = SerperRetriever(serper_config)
            print(f"[Info] SerperRetriever initialized (topk={self.serper_topk})")
            return True

        except Exception as e:
            print(f"[Warning] Failed to initialize SerperRetriever: {e}. Disabling fallback.")
            self.use_serper_fallback = False
            return False

    def _should_trigger_serper_fallback(
        self,
        evaluation: Dict[str, Any],
        docs: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Determine if Serper fallback should be triggered based on review evaluation.

        Args:
            evaluation: The evaluation dict from reviewer response containing problematic_docs
            docs: Current document list

        Returns:
            Tuple of (should_trigger: bool, reason: str)
        """
        if not self.use_serper_fallback:
            return False, "Serper fallback disabled"

        if not docs:
            return False, "No documents available"

        problematic_docs = evaluation.get("problematic_docs", [])
        total_docs = len(docs)
        ratio = len(problematic_docs) / total_docs

        if ratio > self.serper_problematic_ratio_threshold:
            return True, f"Problematic ratio ({ratio:.0%}) > threshold ({self.serper_problematic_ratio_threshold:.0%})"

        return False, "Quality acceptable"

    def _merge_serper_results(
        self,
        local_docs: List[Dict],
        serper_docs: List[Dict],
        docs_to_remove: List[int]
    ) -> List[Dict]:
        """
        Merge Serper search results with local retrieval results.

        Args:
            local_docs: Documents from local retrieval
            serper_docs: Documents from Serper web search
            docs_to_remove: Indices of local docs to remove (problematic)

        Returns:
            Merged document list with deduplication
        """
        # Filter out problematic local documents
        filtered_local = [
            doc for idx, doc in enumerate(local_docs)
            if idx not in docs_to_remove
        ]

        # Normalize Serper results to match local document format
        normalized_serper = []
        for idx, doc in enumerate(serper_docs):
            normalized = {
                "id": doc.get("url", f"serper_{idx}"),
                "contents": doc.get("text", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source": "serper",
            }
            normalized_serper.append(normalized)

        # Deduplicate by URL
        existing_urls = {
            doc.get("url", doc.get("id", ""))
            for doc in filtered_local
        }
        unique_serper = [
            doc for doc in normalized_serper
            if doc.get("url") not in existing_urls
        ]

        # Merge based on strategy
        if self.serper_merge_strategy == "prepend":
            merged = unique_serper + filtered_local
        elif self.serper_merge_strategy == "append":
            merged = filtered_local + unique_serper
        else:
            merged = unique_serper + filtered_local

        # Limit to max_docs
        return merged[:self.max_docs]

    def _batched_generate(self, model, prompts: List[str]) -> List[str]:
        """Generate responses in batches to control memory usage."""
        if len(prompts) <= self.batch_size:
            return model.generate(prompts)

        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            results.extend(model.generate(batch))
        return results

    def format_docs_for_review(self, docs: List[Dict]) -> str:
        """Format documents for reviewer evaluation."""
        formatted = []
        for idx, doc in enumerate(docs):
            content = doc.get("contents", doc.get("content", ""))
            doc_id = doc.get("id", f"doc_{idx}")
            formatted.append(f"[Document {idx}] (ID: {doc_id})\n{content}\n")
        return "\n".join(formatted)

    def format_docs_for_generation(self, docs: List[Dict]) -> str:
        """Format documents for final answer generation."""
        formatted = []
        for idx, doc in enumerate(docs):
            content = doc.get("contents", doc.get("content", ""))
            formatted.append(f"Document {idx + 1}:\n{content}\n")
        return "\n".join(formatted)

    def _safe_doc_id(self, doc: Dict, idx: int) -> str:
        """Return the stable id shown to the reviewer for a document."""
        return str(doc.get("id", f"doc_{idx}"))

    def _extract_json_text(self, response: str) -> str:
        """Extract the first likely JSON object from a model response."""
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()

        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            return response[start:end + 1].strip()

        raise ValueError("No JSON object found in response")

    def _repair_json_text(self, text: str) -> Optional[str]:
        """Apply conservative JSON text repairs before asking the model to retry."""
        if not text:
            return None

        repaired = text.strip()
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        if open_brackets > 0:
            repaired += "]" * open_brackets
        if open_braces > 0:
            repaired += "}" * open_braces

        return repaired if repaired != text else None

    def _parse_json_response(self, response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse a JSON object without converting failures into READY decisions."""
        try:
            parsed = json.loads(self._extract_json_text(response))
            if isinstance(parsed, dict):
                return parsed, None
            return None, "Parsed JSON is not an object"
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            first_error = str(e)

        try:
            json_text = self._extract_json_text(response)
            repaired = self._repair_json_text(json_text)
            if repaired:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    parsed["repaired"] = True
                    parsed["repair_method"] = "deterministic_json_repair"
                    parsed["parse_error"] = first_error
                    parsed["raw_response"] = response
                    return parsed, None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None, first_error

    def _repair_prompt(self, original_prompt: str, bad_response: str, error: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous response was invalid JSON and could not be parsed.\n"
            f"Parse error: {error}\n"
            "Return only one valid JSON object that follows the requested schema.\n"
            "Do not include markdown, explanation, or extra text.\n"
            "Previous invalid response:\n"
            f"{bad_response[:2000]}"
        )

    def _parse_batch_with_retry(self, prompts: List[str], responses: List[str]) -> List[Dict[str, Any]]:
        """
        Parse reviewer responses and retry invalid JSON once in batch.

        Parse failures are surfaced with metadata; downstream logic falls back to
        a conservative answerable state for benchmark QA instead of treating them
        as hard failures.
        """
        parsed_results = []
        retry_indices = []
        retry_prompts = []

        for idx, response in enumerate(responses):
            parsed, error = self._parse_json_response(response)
            if parsed is None:
                parsed_results.append({
                    "parse_failed": True,
                    "failure_reason": "parse_failure",
                    "parse_error": error,
                    "raw_response": response,
                })
                retry_indices.append(idx)
                retry_prompts.append(self._repair_prompt(prompts[idx], response, error or "unknown"))
            else:
                parsed["parse_failed"] = False
                parsed_results.append(parsed)

        if retry_prompts:
            retry_responses = self._batched_generate(self.reviewer, retry_prompts)
            for offset, idx in enumerate(retry_indices):
                parsed, error = self._parse_json_response(retry_responses[offset])
                if parsed is None:
                    parsed_results[idx].update({
                        "retry_failed": True,
                        "retry_error": error,
                        "retry_response": retry_responses[offset],
                    })
                else:
                    parsed["parse_failed"] = False
                    parsed["repaired"] = True
                    parsed["repair_method"] = parsed.get("repair_method") or "model_json_retry"
                    parsed["parse_error"] = parsed_results[idx].get("parse_error")
                    parsed["raw_response"] = parsed_results[idx].get("raw_response")
                    parsed_results[idx] = parsed

        return parsed_results

    def _truncate_text(self, text: Optional[str], max_chars: int = 2000) -> Optional[str]:
        if text is None:
            return None
        text = str(text)
        return text if len(text) <= max_chars else text[:max_chars]

    def _doc_title(self, doc: Dict) -> str:
        title = doc.get("title")
        if title:
            return str(title)
        content = str(doc.get("contents", doc.get("content", ""))).strip()
        if not content:
            return ""
        first_line = content.splitlines()[0].strip()
        quoted = re.match(r'^"([^"]+)"', first_line)
        if quoted:
            return quoted.group(1).strip()
        return first_line[:120]

    def _doc_score(self, doc: Dict) -> Optional[float]:
        for key in ("score", "retrieval_score", "retriever_score"):
            if key in doc:
                try:
                    return float(doc[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _compact_docs(self, docs: List[Dict]) -> List[Dict[str, Any]]:
        compact = []
        for idx, doc in enumerate(docs):
            compact.append({
                "id": self._safe_doc_id(doc, idx),
                "title": self._doc_title(doc),
                "score": self._doc_score(doc),
            })
        return compact

    def _empty_diagnostic_trace(self) -> Dict[str, Any]:
        return {
            "pipeline": {
                "pipeline_version": self.pipeline_version,
                "state_schema_version": self.state_schema_version,
                "prompt_version": self.prompt_version,
                "enable_state_machine_executor": bool(self.enable_state_machine_executor),
                "enable_structured_retrieval_plan": self.enable_structured_retrieval_plan,
                "enable_answer_verifier": self.enable_answer_verifier,
                "enable_hybrid_retrieval": self.enable_hybrid_retrieval,
                "enable_relation_grounding": self.enable_relation_grounding,
                "relation_grounding_mode": self.relation_grounding_mode,
                "enable_comparison_executor": self.enable_comparison_executor,
                "enable_comparison_output_contract": False,
                "enable_comparison_output_contract_diagnostics": True,
                "comparison_branch_mode": self.comparison_branch_mode,
                "comparison_finalization_mode": self.comparison_finalization_mode,
                "deterministic_commit_scope": self.deterministic_commit_scope,
                "enable_deterministic_answer_commit": self.enable_deterministic_answer_commit,
            },
            "question_analysis": {
                "question_type": "",
                "original_question_type": "",
                "effective_question_type": "",
                "question_type_repaired": False,
                "question_type_repair_reason": None,
                "slot_count": 0,
                "slot_descriptions": [],
                "parse_failed": False,
                "parse_error": None,
                "raw_response_on_failure": None,
                "repaired": False,
            },
            "review_trace": [],
            "retrieval_trace": [],
            "ranking_trace": [],
            "chain_grounding_trace": {},
            "relation_grounding_trace": {},
            "answer_decision": {},
            "gold_evidence_trace": {},
            "verifier_trace": {},
            "final_trace": {},
        }

    def _attach_policy_diagnostics(
        self,
        diagnostic_trace: Dict[str, Any],
        policy: Optional[Dict[str, Any]],
    ) -> None:
        policy = policy or {}
        relation_trace = policy.get("relation_grounding_trace") or {}
        answer_decision = policy.get("answer_decision") or {}
        diagnostic_trace["chain_grounding_trace"] = policy.get("chain_grounding_trace", {})
        diagnostic_trace["relation_grounding_trace"] = relation_trace
        diagnostic_trace["answer_decision"] = answer_decision
        for key in (
            "comparison_output_contract",
            "comparison_output_contract_available",
            "comparison_output_contract_applied",
            "comparison_finalization_mode",
            "comparison_commit_eligible",
            "comparison_commit_blocked_reason",
            "comparison_commit_confidence",
            "comparison_commit_relation_family",
            "comparison_commit_relation_value_type_valid",
            "comparison_slot_associated_rejected",
            "comparison_final_candidate_matches_option",
            "comparison_guided_generation",
            "comparison_final_commit_saved_generation",
        ):
            diagnostic_trace[key] = policy.get(key)
        for key in (
            "relation_grounded",
            "semantic_coverage",
            "subject_grounded",
            "target_relation_match",
            "candidate_role_valid",
            "candidate_answer_type_match",
            "candidate_is_question_entity",
            "candidate_is_intermediate_entity",
            "grounded_but_wrong_relation",
            "comparison_execution_status",
            "intent_source",
            "intent_conflict",
            "operand_binding_status",
            "comparison_lineage",
            "strict_comparison_executable",
            "counterfactual_answer_candidate",
            "counterfactual_candidate_source",
            "would_commit_under_v2_7",
            "generated_without_validated_candidate",
            "entity_binding_valid",
            "binding_directions",
        ):
            diagnostic_trace[key] = relation_trace.get(key)
        diagnostic_trace["control_action"] = relation_trace.get("control_action", "diagnostic_only")
        diagnostic_trace["forced_by_relation_guard"] = bool(
            (policy.get("failure_reason") or "") == "relation_ungrounded_final_slot"
        )
        diagnostic_trace["generated_without_validated_candidate"] = bool(
            policy.get("generated_without_validated_candidate")
        )
        diagnostic_trace["semantic_guidance_applied"] = bool(policy.get("semantic_guidance_applied"))
        diagnostic_trace["diagnostic_candidate_leakage"] = bool(policy.get("diagnostic_candidate_leakage"))
        diagnostic_trace["commit_mode"] = answer_decision.get("commit_mode")
        diagnostic_trace["commit_reason"] = answer_decision.get("commit_reason")
        diagnostic_trace["committed_claim_ids"] = answer_decision.get("committed_claim_ids") or []

    def _config_value(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config[key]
        except Exception:
            return default
        return default if value is None else value

    def _safe_git_commit(self) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        commit = completed.stdout.strip()
        return commit or None

    def _safe_git_dirty(self) -> Optional[bool]:
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain", "--", "examples/reviewer"],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        return bool(completed.stdout.strip()) if completed.returncode == 0 else None

    def _code_fingerprint(self) -> Optional[str]:
        base_dir = os.path.dirname(__file__)
        filenames = (
            "reviewer_state.py", "reviewer_answer_semantics.py", "reviewer_pipeline.py",
            "run_reviewer.py", "run_all.sh", "reviewer_config.yaml",
            "analyze_reviewer_results.py",
        )
        digest = hashlib.sha256()
        try:
            for filename in filenames:
                path = os.path.join(base_dir, filename)
                digest.update(filename.encode("utf-8"))
                with open(path, "rb") as fp:
                    digest.update(fp.read())
        except OSError:
            return None
        return digest.hexdigest()

    def _build_manifest(self, dataset_size: Optional[int] = None) -> Dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "state_schema_version": self.state_schema_version,
            "prompt_version": self.prompt_version,
            "git_commit": self._safe_git_commit(),
            "git_dirty": self._safe_git_dirty(),
            "code_fingerprint": self._code_fingerprint(),
            "dataset_name": self._config_value("dataset_name"),
            "split": self._config_value("split"),
            "dataset_size": dataset_size,
            "save_note": self._config_value("save_note"),
            "result_tag": self._config_value("result_tag"),
            "max_iterations": self.max_iterations,
            "model_roles": {
                "generator_model": self._config_value("generator_model"),
                "reviewer_model": "shared_generator" if self.reviewer is self.generator else "separate_reviewer",
            },
            "retrieval": {
                "retrieval_method": self._config_value("retrieval_method"),
                "retrieval_topk": self._config_value("retrieval_topk"),
                "max_slot_queries": self.max_slot_queries,
                "max_docs": self.max_docs,
                "final_max_docs": self.final_max_docs,
            },
            "feature_flags": {
                "enable_state_machine_executor": bool(self.enable_state_machine_executor),
                "enable_structured_retrieval_plan": self.enable_structured_retrieval_plan,
                "enable_answer_verifier": self.enable_answer_verifier,
                "enable_hybrid_retrieval": self.enable_hybrid_retrieval,
                "enable_relation_grounding": self.enable_relation_grounding,
                "relation_grounding_mode": self.relation_grounding_mode,
                "enable_comparison_executor": self.enable_comparison_executor,
                "enable_comparison_output_contract": False,
                "enable_comparison_output_contract_diagnostics": True,
                "comparison_branch_mode": self.comparison_branch_mode,
                "comparison_finalization_mode": self.comparison_finalization_mode,
                "deterministic_commit_scope": self.deterministic_commit_scope,
                "enable_deterministic_answer_commit": self.enable_deterministic_answer_commit,
            },
        }

    def _write_manifest(self, dataset_size: Optional[int] = None) -> None:
        save_dir = self._config_value("save_dir")
        if not save_dir:
            return
        manifest_path = os.path.join(save_dir, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(self._build_manifest(dataset_size), fp, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"[Warning] Failed to save reviewer manifest: {exc}")

    def _question_analysis_trace(self, state: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        slots = state.get("answer_slots", [])
        raw_response = None
        if parsed.get("parse_failed") or parsed.get("retry_failed") or parsed.get("repaired"):
            raw_response = parsed.get("raw_response") or parsed.get("retry_response")
        return {
            "question_type": state.get("question_type", ""),
            "original_question_type": state.get("original_question_type", state.get("question_type", "")),
            "effective_question_type": state.get("question_type", ""),
            "question_type_repaired": bool(state.get("question_type_repaired")),
            "question_type_repair_reason": state.get("question_type_repair_reason"),
            "answer_type": state.get("answer_type"),
            "answer_intent": state.get("answer_intent", {}),
            "slot_count": len(slots),
            "slot_descriptions": [slot.get("description", "") for slot in slots],
            "slot_roles": [slot.get("slot_role", "") for slot in slots],
            "parse_failed": bool(parsed.get("parse_failed") or parsed.get("retry_failed")),
            "parse_error": parsed.get("parse_error") or parsed.get("retry_error"),
            "raw_response_on_failure": self._truncate_text(raw_response),
            "repaired": bool(parsed.get("repaired")),
            "repair_method": parsed.get("repair_method"),
        }

    def _review_step_trace(
        self,
        iteration: int,
        num_docs_before_review: int,
        parsed: Dict[str, Any],
        state: Dict[str, Any],
        coverage: Dict[str, Any],
        new_claim_count: int,
        decision_before_fallback: str,
    ) -> Dict[str, Any]:
        raw_response = None
        if parsed.get("parse_failed") or parsed.get("retry_failed") or parsed.get("repaired"):
            raw_response = parsed.get("raw_response") or parsed.get("retry_response")
        retrieval_plan = state.get("retrieval_plan", [])
        return {
            "iteration": iteration,
            "num_docs_before_review": num_docs_before_review,
            "decision_before_fallback": decision_before_fallback,
            "decision_after_merge": state.get("decision"),
            "slot_coverage": coverage,
            "new_claim_count": new_claim_count,
            "claim_count_total": len(state.get("claims", [])),
            "retrieval_plan_count": len(retrieval_plan),
            "retrieval_plan": retrieval_plan,
            "docs_to_remove": state.get("docs_to_remove", []),
            "ignored_docs": state.get("ignored_docs", []),
            "direct_answer_supported": bool(state.get("direct_answer_supported")),
            "answer_type": state.get("answer_type"),
            "answer_candidate": state.get("answer_candidate"),
            "ready_guard_triggered": bool(state.get("ready_guard_triggered")),
            "guard_extra_round_pending": bool(state.get("guard_extra_round_pending")),
            "guard_extra_round_used": bool(state.get("guard_extra_round_used")),
            "failure_reason": state.get("failure_reason"),
            "state_consistency_issues": state.get("state_consistency_issues", []),
            "chain_grounding_trace": state.get("chain_grounding_trace", {}),
            "chain_grounded": bool((state.get("chain_grounding_trace") or {}).get("chain_grounded")),
            "chain_coverage": (state.get("chain_grounding_trace") or {}).get("chain_coverage"),
            "relation_grounding_trace": state.get("relation_grounding_trace", {}),
            "answer_decision": state.get("answer_decision", {}),
            "parse_failed": bool(parsed.get("parse_failed") or parsed.get("retry_failed")),
            "parse_error": parsed.get("parse_error") or parsed.get("retry_error"),
            "repaired": bool(parsed.get("repaired")),
            "repair_method": parsed.get("repair_method"),
            "raw_response_on_failure": self._truncate_text(raw_response),
        }

    def _infer_question_type(self, question: str) -> str:
        q = question.lower()
        comparison_markers = [
            "both", "same", "first", "earlier", "later", "older", "younger",
            "larger", "smaller", "more", "less", "compare", "most recently",
            "most recent", "latest", "newer", "newest", "oldest", "highest",
            "lowest", "longer", "shorter", "fewer",
        ]
        bridge_markers = ["who", "what", "where", "when", "which"]
        if any(marker in q for marker in comparison_markers):
            return "comparison"
        if any(marker in q for marker in bridge_markers) and (" of " in q or " by " in q or " in " in q):
            return "bridge"
        return "single_hop"

    def _infer_answer_type(self, question: str) -> str:
        q = str(question or "").lower().strip()
        if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")):
            return "yes_no"
        if re.search(r"\b(how many|how much|what number|what year|what percentage|what percent)\b", q):
            return "number"
        if re.search(r"\b(when|what date|which year|what year)\b", q):
            return "date"
        if re.search(r"\b(where|what city|what country|which country|which city|location)\b", q):
            return "location"
        if re.search(r"\b(who|whose|person|actor|actress|singer|writer|director|president|player|composer)\b", q):
            return "person"
        if re.search(r"\b(which|what film|what movie|what book|what novel|what song|what album|what series)\b", q):
            return "title_or_entity"
        return "phrase"

    def _fallback_slots(self, question: str, question_type: Optional[str] = None) -> List[Dict[str, Any]]:
        question_type = question_type or self._infer_question_type(question)
        if question_type == "comparison":
            slot_specs = [
                ("comparison_operand", [], "Identify the first entity and its relevant attribute."),
                ("comparison_operand", [], "Identify the second entity and its relevant attribute."),
                ("final_answer", ["slot_1", "slot_2"], "Compare the attributes to answer the question."),
            ]
        elif question_type == "bridge":
            slot_specs = [
                ("intermediate_entity", [], "Identify the intermediate entity or relation required by the question."),
                ("final_answer", ["slot_1"], "Use the intermediate entity or relation to determine the final answer."),
            ]
        else:
            slot_specs = [("direct_answer", [], "Find the direct fact that answers the question.")]

        return [
            {
                "slot_id": f"slot_{idx + 1}",
                "description": desc,
                "slot_role": slot_role,
                "depends_on": depends_on,
                "status": "missing",
                "supporting_claim_ids": [],
                "conflicting_claim_ids": [],
            }
            for idx, (slot_role, depends_on, desc) in enumerate(slot_specs)
        ]

    def _normalize_slots(self, slots: Any, question: str, question_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(slots, list) or not slots:
            return repair_answer_slot_topology(
                self._fallback_slots(question, question_type), question, question_type
            )

        normalized = []
        seen = set()
        for idx, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            slot_id = str(slot.get("slot_id") or f"slot_{idx + 1}")
            if slot_id in seen:
                slot_id = f"slot_{idx + 1}"
            seen.add(slot_id)
            status = slot.get("status") or "missing"
            if status not in {"missing", "partial", "supported", "conflicted"}:
                status = "missing"
            normalized.append({
                "slot_id": slot_id,
                "description": str(slot.get("description") or f"Required evidence slot {idx + 1}"),
                "slot_role": str(slot.get("slot_role") or self._default_slot_role(question_type, idx, len(slots))),
                "depends_on": list(slot.get("depends_on") or []),
                "status": status,
                "supporting_claim_ids": list(slot.get("supporting_claim_ids") or []),
                "conflicting_claim_ids": list(slot.get("conflicting_claim_ids") or []),
                "target_relation": slot.get("target_relation"),
                "relation_family": slot.get("relation_family"),
                "expected_answer_type": slot.get("expected_answer_type"),
                "comparison_option": slot.get("comparison_option"),
            })

        normalized = normalized or self._fallback_slots(question, question_type)
        return repair_answer_slot_topology(normalized, question, question_type)

    def _default_slot_role(self, question_type: Optional[str], idx: int, slot_count: int) -> str:
        if question_type == "bridge":
            return "final_answer" if idx == slot_count - 1 else "intermediate_entity"
        if question_type == "comparison":
            return "final_answer" if idx == slot_count - 1 else "comparison_operand"
        return "direct_answer"

    def _coerce_evidence_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize EvidenceState at pipeline boundaries while keeping dict output."""
        return normalize_review_state(state)

    def _normalize_question_analysis(self, parsed: Dict[str, Any], question: str) -> Dict[str, Any]:
        valid_types = {"single_hop", "bridge", "comparison", "ambiguous"}
        original_question_type = parsed.get("question_type")
        if original_question_type not in valid_types:
            original_question_type = self._infer_question_type(question)

        slots = self._normalize_slots(parsed.get("answer_slots"), question, original_question_type)
        question_type, repair_reason = infer_effective_question_type(
            original_question_type, slots, question
        )
        slots = repair_answer_slot_topology(slots, question, question_type)
        answer_intent = build_answer_intent(
            question,
            question_type,
            self._infer_answer_type(question),
            slots,
            parsed.get("answer_intent"),
        )
        slots = annotate_slots_with_intent(slots, answer_intent)
        state = {
            "question_type": question_type,
            "original_question_type": original_question_type,
            "question_type_repaired": question_type != original_question_type,
            "question_type_repair_reason": repair_reason,
            "answer_slots": slots,
            "claims": [],
            "retrieval_plan": [],
            "decision": "continue_retrieval",
            "answer_type": answer_intent.get("expected_answer_type") or self._infer_answer_type(question),
            "answer_intent": answer_intent,
            "direct_answer_supported": False,
            "answer_candidate": None,
            "answer_candidate_claim_ids": [],
            "failure_reason": parsed.get("failure_reason"),
            "explanation": parsed.get("explanation", ""),
            "round_id": 0,
            "comparison_view_applied": False,
            "derived_slots_hidden_from_reviewer": 0,
            "derived_retrieval_plans_dropped": 0,
            "comparison_branch_executable": False,
            "comparison_fallback_to_patch2": question_type == "comparison",
        }
        state["comparison_output_contract"] = build_comparison_output_contract(question, state)
        return self._coerce_evidence_state(state)

    def _normalize_claims(self, claims: Any, valid_slot_ids: set, round_id: int) -> List[Dict[str, Any]]:
        if not isinstance(claims, list):
            return []

        normalized = []
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text") or "").strip()
            if not text:
                continue
            slot_id = str(claim.get("slot_id") or "")
            if slot_id not in valid_slot_ids and valid_slot_ids:
                slot_id = sorted(valid_slot_ids)[0]
            support_status = claim.get("support_status") or claim.get("relation_to_question") or "support"
            if support_status not in {"support", "contradict", "irrelevant"}:
                support_status = "support"
            semantic_values = {}
            for key in ("subject", "predicate", "object", "object_type"):
                value = claim.get(key)
                if value is None or str(value).strip().lower() in {"", "null", "none", "unknown", "n/a"}:
                    semantic_values[key] = None
                else:
                    semantic_values[key] = str(value).strip()
            normalized.append({
                "claim_id": str(claim.get("claim_id") or f"c{idx + 1}"),
                "doc_id": str(claim.get("doc_id") or ""),
                "slot_id": slot_id,
                "text": text,
                "support_status": support_status,
                "round_id": claim.get("round_id", round_id),
                **semantic_values,
            })
        return normalized

    def _normalize_retrieval_plan(self, plan: Any, valid_slot_ids: set) -> List[Dict[str, Any]]:
        if not isinstance(plan, list):
            return []

        normalized = []
        seen_queries = set()
        for idx, item in enumerate(plan):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query or query.lower() in seen_queries:
                continue
            slot_id = str(item.get("slot_id") or "")
            if slot_id not in valid_slot_ids and valid_slot_ids:
                slot_id = sorted(valid_slot_ids)[0]
            normalized.append({
                "slot_id": slot_id,
                "query": query,
                "reason": str(item.get("reason") or "Need more evidence for this slot."),
            })
            seen_queries.add(query.lower())
            if len(normalized) >= self.max_slot_queries:
                break
        return normalized

    def _normalize_slot_updates(self, updates: Any, previous_slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        slots = json.loads(json.dumps(previous_slots or []))
        if not isinstance(updates, list):
            return slots

        slots_by_id = {str(slot.get("slot_id")): slot for slot in slots}
        valid_statuses = {"missing", "partial", "supported", "conflicted"}
        for update in updates:
            if not isinstance(update, dict):
                continue
            slot_id = str(update.get("slot_id") or "")
            slot = slots_by_id.get(slot_id)
            if not slot:
                continue
            status = update.get("status")
            if status in valid_statuses:
                slot["status"] = status
            if "supporting_claim_ids" in update:
                slot["supporting_claim_ids"] = list(update.get("supporting_claim_ids") or [])
            if "conflicting_claim_ids" in update:
                slot["conflicting_claim_ids"] = list(update.get("conflicting_claim_ids") or [])
        return slots

    def _final_answer_slot_ids(self, slots: List[Dict[str, Any]]) -> set:
        return {
            slot.get("slot_id")
            for slot in slots
            if slot.get("slot_role") in {"final_answer", "direct_answer"}
        }

    def _build_bridge_fallback_plan(
        self,
        question: str,
        slots: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        final_slots = [
            slot for slot in slots
            if slot.get("slot_role") == "final_answer"
        ]
        if not final_slots:
            return []

        intermediate_ids = {
            slot.get("slot_id")
            for slot in slots
            if slot.get("slot_role") == "intermediate_entity"
        }
        intermediate_claims = [
            claim.get("text", "").strip()
            for claim in claims
            if claim.get("slot_id") in intermediate_ids
            and claim.get("support_status") == "support"
            and claim.get("text")
        ]
        if not intermediate_claims:
            return []

        intermediate_claims.sort(key=lambda text: (len(text.split()), len(text)))
        final_slot = final_slots[0]
        intermediate_entity = self._claim_to_short_answer_candidate(intermediate_claims[0]) or intermediate_claims[0]
        query = f"{intermediate_entity} {final_slot.get('description', '')}".strip()
        return [{
            "slot_id": final_slot.get("slot_id"),
            "query": query,
            "reason": "Retrieve the final answer attribute using the resolved intermediate entity.",
        }]

    def _merge_evidence_state(
        self,
        previous_state: Dict[str, Any],
        parsed: Dict[str, Any],
        question: str,
        round_id: int,
    ) -> Dict[str, Any]:
        if parsed.get("parse_failed"):
            state = json.loads(json.dumps(previous_state))
            state["decision"] = "ready_to_answer" if state.get("direct_answer_supported") else "force_answer"
            state["failure_reason"] = "parse_failure_fallback"
            state["retrieval_plan"] = []
            state["round_id"] = round_id
            return self._coerce_evidence_state(state)

        if "answer_slots" in parsed:
            slots = self._normalize_slots(
                parsed.get("answer_slots", previous_state.get("answer_slots")),
                question,
                previous_state.get("question_type"),
            )
        else:
            slots = self._normalize_slot_updates(
                parsed.get("slot_updates"),
                previous_state.get("answer_slots", []),
            )
        answer_intent = build_answer_intent(
            question,
            previous_state.get("question_type", self._infer_question_type(question)),
            previous_state.get("answer_type", self._infer_answer_type(question)),
            slots,
            previous_state.get("answer_intent"),
        )
        slots = annotate_slots_with_intent(slots, answer_intent)
        valid_slot_ids = {slot["slot_id"] for slot in slots}
        evidence_slot_ids = self._evidence_slot_ids({
            "question_type": previous_state.get("question_type"),
            "answer_slots": slots,
            "comparison_branch_executable": previous_state.get("comparison_branch_executable"),
        })

        old_claims = previous_state.get("claims", [])
        new_claims = self._normalize_claims(parsed.get("claims"), valid_slot_ids, round_id)
        claim_signatures = {
            (claim.get("doc_id"), claim.get("slot_id"), claim.get("text", "").lower())
            for claim in old_claims
        }
        merged_claims = list(old_claims)
        for claim in new_claims:
            signature = (claim.get("doc_id"), claim.get("slot_id"), claim.get("text", "").lower())
            if signature not in claim_signatures:
                merged_claims.append(claim)
                claim_signatures.add(signature)

        decision = parsed.get("decision")
        if decision == "READY_TO_ANSWER":
            decision = "ready_to_answer"
        elif decision == "NEED_MORE_INFO":
            decision = "continue_retrieval"
        if decision not in {"continue_retrieval", "ready_to_answer", "force_answer", "review_failed"}:
            decision = "continue_retrieval"

        raw_retrieval_plan = parsed.get("retrieval_plan")
        derived_retrieval_plans_dropped = 0
        if (
            self._comparison_branch_control_active(previous_state)
            and isinstance(raw_retrieval_plan, list)
        ):
            original_plan_count = len(raw_retrieval_plan)
            raw_retrieval_plan = [
                plan for plan in raw_retrieval_plan
                if isinstance(plan, dict)
                and str(plan.get("slot_id") or "") in evidence_slot_ids
            ]
            derived_retrieval_plans_dropped = original_plan_count - len(raw_retrieval_plan)
        retrieval_plan = self._normalize_retrieval_plan(
            raw_retrieval_plan, evidence_slot_ids or valid_slot_ids
        )
        final_slots = [
            slot for slot in slots
            if slot.get("slot_role") in {"final_answer", "direct_answer"}
        ]
        if "direct_answer_supported" in parsed:
            direct_answer_supported = bool(parsed.get("direct_answer_supported"))
        else:
            direct_answer_supported = bool(final_slots) and all(
                slot.get("status") == "supported" for slot in final_slots
            )
        if self._comparison_branch_control_active(previous_state):
            operand_slots = [slot for slot in slots if slot.get("slot_role") == "comparison_operand"]
            direct_answer_supported = bool(len(operand_slots) >= 2) and all(
                slot.get("status") == "supported" for slot in operand_slots
            )
        answer_candidate = str(parsed.get("answer_candidate") or "").strip() or None
        answer_candidate_claim_ids = list(parsed.get("answer_candidate_claim_ids") or [])
        valid_answer_types = {"yes_no", "person", "date", "location", "title_or_entity", "number", "phrase"}
        answer_type = answer_intent.get("expected_answer_type") or previous_state.get("answer_type") or self._infer_answer_type(question)
        if answer_type not in valid_answer_types:
            answer_type = self._infer_answer_type(question)
        final_answer_slot_ids = self._final_answer_slot_ids(slots)
        candidate_claim_ids = {
            claim.get("claim_id")
            for claim in merged_claims
            if claim.get("slot_id") in final_answer_slot_ids
            and claim.get("support_status") == "support"
        }
        if (
            answer_candidate
            and answer_candidate_claim_ids
            and not self._comparison_branch_control_active(previous_state)
        ):
            direct_answer_supported = bool(candidate_claim_ids.intersection(answer_candidate_claim_ids))

        ready_guard_triggered = False
        guard_extra_round_pending = bool(previous_state.get("guard_extra_round_pending"))
        if decision == "ready_to_answer" and not direct_answer_supported:
            ready_guard_triggered = True
            if not retrieval_plan and previous_state.get("question_type") == "bridge":
                retrieval_plan = self._build_bridge_fallback_plan(question, slots, merged_claims)
            if retrieval_plan and not previous_state.get("guard_extra_round_used"):
                guard_extra_round_pending = True
            decision = "continue_retrieval" if retrieval_plan else "force_answer"

        if decision == "continue_retrieval" and not retrieval_plan:
            unresolved = [
                slot for slot in slots
                if slot.get("slot_id") in evidence_slot_ids
                and slot["status"] in {"missing", "partial", "conflicted"}
            ]
            if not unresolved:
                decision = "ready_to_answer" if direct_answer_supported else "force_answer"
            else:
                if previous_state.get("question_type") == "bridge":
                    retrieval_plan = self._build_bridge_fallback_plan(question, slots, merged_claims)
                decision = "continue_retrieval" if retrieval_plan else "force_answer"

        state = {
            "question_type": previous_state.get("question_type", self._infer_question_type(question)),
            "answer_slots": slots,
            "claims": merged_claims,
            "retrieval_plan": retrieval_plan,
            "decision": decision,
            "answer_type": answer_type,
            "answer_intent": answer_intent,
            "comparison_output_contract": build_comparison_output_contract(question, {
                "question_type": previous_state.get("question_type"),
                "answer_type": answer_type,
                "answer_slots": slots,
                "answer_intent": answer_intent,
            }),
            "relation_grounding_trace": previous_state.get("relation_grounding_trace", {}),
            "answer_decision": previous_state.get("answer_decision", {}),
            "direct_answer_supported": direct_answer_supported,
            "answer_candidate": answer_candidate,
            "answer_candidate_claim_ids": answer_candidate_claim_ids,
            "ready_guard_triggered": bool(previous_state.get("ready_guard_triggered")) or ready_guard_triggered,
            "guard_extra_round_pending": guard_extra_round_pending,
            "guard_extra_round_used": bool(previous_state.get("guard_extra_round_used")),
            "docs_to_remove": self._sanitize_doc_indices(parsed.get("docs_to_remove", [])),
            "ignored_docs": self._sanitize_doc_indices(parsed.get("ignored_docs", [])),
            "failure_reason": parsed.get("failure_reason"),
            "explanation": parsed.get("explanation", ""),
            "round_id": round_id,
            "comparison_view_applied": self._comparison_branch_control_active(previous_state),
            "derived_slots_hidden_from_reviewer": sum(
                1 for slot in slots
                if slot.get("slot_role") == "final_answer" and slot.get("derived_output")
            ) if self._comparison_branch_control_active(previous_state) else 0,
            "derived_retrieval_plans_dropped": int(
                previous_state.get("derived_retrieval_plans_dropped") or 0
            ) + derived_retrieval_plans_dropped,
            "comparison_branch_executable": bool(
                previous_state.get("comparison_branch_executable")
            ),
            "comparison_fallback_to_patch2": bool(
                previous_state.get("question_type") == "comparison"
                and not self._comparison_branch_control_active(previous_state)
            ),
        }
        state, consistency_issues = repair_slot_claim_consistency(state, question=question)
        state["state_consistency_issues"] = consistency_issues
        if consistency_issues:
            comparison_view_applied = self._comparison_branch_control_active(state)
            final_slots = [
                slot for slot in state.get("answer_slots", [])
                if slot.get("slot_role") in {"final_answer", "direct_answer"}
            ]
            final_answer_slot_ids = self._final_answer_slot_ids(state.get("answer_slots", []))
            candidate_claim_ids = {
                claim.get("claim_id")
                for claim in state.get("claims", [])
                if claim.get("slot_id") in final_answer_slot_ids
                and claim.get("support_status") == "support"
            }
            if comparison_view_applied:
                operand_slots = [
                    slot for slot in state.get("answer_slots", [])
                    if slot.get("slot_role") == "comparison_operand"
                ]
                state["direct_answer_supported"] = bool(len(operand_slots) >= 2) and all(
                    slot.get("status") == "supported" for slot in operand_slots
                )
            elif state.get("answer_candidate") and state.get("answer_candidate_claim_ids"):
                state["direct_answer_supported"] = bool(
                    candidate_claim_ids.intersection(state.get("answer_candidate_claim_ids") or [])
                )
            else:
                state["direct_answer_supported"] = bool(final_slots) and all(
                    slot.get("status") == "supported" for slot in final_slots
                )
            unresolved = [
                slot for slot in state.get("answer_slots", [])
                if slot.get("slot_id") in self._evidence_slot_ids(state)
                and slot.get("status") in {"missing", "partial", "conflicted"}
            ]
            if state.get("decision") == "ready_to_answer":
                has_candidate = comparison_view_applied and state.get("direct_answer_supported") or bool(
                    state.get("answer_candidate") or state.get("answer_candidate_claim_ids")
                )
                if unresolved or not state.get("direct_answer_supported") or not has_candidate:
                    state["ready_guard_triggered"] = True
                    if state.get("retrieval_plan") and not state.get("guard_extra_round_used"):
                        state["guard_extra_round_pending"] = True
                        state["decision"] = "continue_retrieval"
                    else:
                        state["decision"] = "force_answer"
        return self._coerce_evidence_state(state)

    def _sanitize_doc_indices(self, value: Any) -> List[int]:
        if not isinstance(value, list):
            return []
        sanitized = []
        for item in value:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx >= 0:
                sanitized.append(idx)
        return sanitized

    def _comparison_branch_control_active(self, state: Dict[str, Any]) -> bool:
        return bool(
            getattr(self, "enable_comparison_executor", True)
            and getattr(self, "comparison_branch_mode", "diagnostic") == "control"
            and state.get("question_type") == "comparison"
            and state.get("comparison_branch_executable")
        )

    def _slot_coverage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("question_type") == "comparison" and not self._comparison_branch_control_active(state):
            slots = ReviewState.from_dict(state).answer_slots
            total = len(slots)
            supported = sum(slot.status.value == "supported" for slot in slots)
            conflicted = sum(slot.status.value == "conflicted" for slot in slots)
            missing = total - supported - conflicted
            return {
                "total": total,
                "supported": supported,
                "conflicted": conflicted,
                "missing": missing,
                "coverage": supported / total if total else 0.0,
            }
        return ReviewState.from_dict(state).slot_coverage()

    def _evidence_slot_ids(self, state: Dict[str, Any]) -> set:
        slots = state.get("answer_slots") or []
        if state.get("question_type") != "comparison" or not self._comparison_branch_control_active(state):
            return {slot.get("slot_id") for slot in slots if slot.get("slot_id")}
        return {
            slot.get("slot_id") for slot in slots
            if slot.get("slot_id")
            and slot.get("slot_role") != "final_answer"
            and not slot.get("derived_output")
        }

    def _build_chain_grounding_trace(
        self,
        state: Dict[str, Any],
        docs: Optional[List[Dict]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        slots = state.get("answer_slots") or []
        claims = state.get("claims") or []
        final_slot_ids = self._final_answer_slot_ids(slots)
        if not final_slot_ids and state.get("question_type") == "comparison":
            final_slot_ids = {
                slot.get("slot_id")
                for slot in slots
                if slot.get("status") == "supported"
            }
        intermediate_slot_ids = {
            slot.get("slot_id")
            for slot in slots
            if slot.get("slot_id") not in final_slot_ids
            and slot.get("slot_role") not in {"final_answer", "direct_answer"}
        }
        title_by_doc_id = {}
        for idx, doc in enumerate(docs or []):
            doc_id = self._safe_doc_id(doc, idx)
            title = self._doc_title(doc)
            title_by_doc_id[str(doc_id)] = title
            if doc.get("id") is not None:
                title_by_doc_id[str(doc.get("id"))] = title

        resolved_entities = []
        seen_entities = set()
        for claim in claims:
            if claim.get("slot_id") not in intermediate_slot_ids:
                continue
            if claim.get("support_status") != "support":
                continue
            for entity in self._extract_entities_from_claim(claim.get("text", "")):
                key = self._normalize_entity(entity)
                if not key or key in seen_entities:
                    continue
                seen_entities.add(key)
                resolved_entities.append({
                    "entity": entity,
                    "normalized": key,
                    "slot_id": claim.get("slot_id"),
                    "claim_id": claim.get("claim_id"),
                })

        final_support_claims = [
            claim for claim in claims
            if claim.get("slot_id") in final_slot_ids
            and claim.get("support_status") == "support"
        ]
        grounded_ids = []
        ungrounded_ids = []
        claim_grounding = {}
        has_intermediate_slots = bool(intermediate_slot_ids)
        for claim in final_support_claims:
            claim_id = claim.get("claim_id")
            if not has_intermediate_slots:
                grounded = True
                matched_entities = []
                reason = "direct_answer"
            elif not resolved_entities:
                grounded = False
                matched_entities = []
                reason = "no_resolved_intermediate_entity"
            else:
                claim_text = str(claim.get("text") or "")
                doc_title = title_by_doc_id.get(str(claim.get("doc_id") or ""), "")
                haystack = f"{claim_text} {doc_title}"
                matched_entities = [
                    entity["entity"]
                    for entity in resolved_entities
                    if self._entity_mentioned(entity["entity"], haystack)
                ]
                grounded = bool(matched_entities)
                reason = "matched_intermediate_entity" if grounded else "final_claim_not_bound_to_intermediate_entity"
            claim_grounding[str(claim_id)] = {
                "grounded": grounded,
                "matched_entities": matched_entities,
                "reason": reason,
            }
            if grounded:
                if claim_id is not None:
                    grounded_ids.append(claim_id)
            else:
                if claim_id is not None:
                    ungrounded_ids.append(claim_id)

        intermediate_supported = (
            not has_intermediate_slots
            or bool(resolved_entities)
        )
        final_supported = bool(final_support_claims)
        final_grounded = bool(grounded_ids) if final_supported else False
        if not has_intermediate_slots:
            chain_coverage = 1.0 if final_supported else 0.0
        else:
            chain_coverage = (
                int(intermediate_supported)
                + int(final_supported)
                + int(final_grounded)
            ) / 3.0
        if not final_supported:
            ungrounded_reason = "no_final_support_claim"
        elif not intermediate_supported:
            ungrounded_reason = "no_resolved_intermediate_entity"
        elif not final_grounded:
            ungrounded_reason = "final_claim_not_bound_to_intermediate_entity"
        else:
            ungrounded_reason = None
        return {
            "chain_grounded": bool(final_grounded),
            "chain_coverage": chain_coverage,
            "resolved_intermediate_entities": resolved_entities,
            "grounded_final_claim_ids": grounded_ids,
            "ungrounded_final_claim_ids": ungrounded_ids,
            "claim_grounding": claim_grounding,
            "ungrounded_reason": ungrounded_reason,
            "requires_grounding": has_intermediate_slots,
        }

    def _build_relation_grounding_trace(
        self,
        state: Dict[str, Any],
        docs: Optional[List[Dict]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        state_before = json.dumps(state, ensure_ascii=False, sort_keys=True)
        trace = evaluate_answer_semantics(
            str(question or ""),
            state,
            docs=docs,
            candidate_extractor=self._claim_to_answer_candidate,
            answer_type_matcher=self._candidate_matches_answer_type,
            enable_comparison_executor=self.enable_comparison_executor,
            deterministic_commit_scope=(
                self.deterministic_commit_scope
                if self.enable_deterministic_answer_commit
                and self.relation_grounding_mode == "control"
                else "none"
            ),
        )
        trace["diagnostic_state_mutated"] = state_before != json.dumps(
            state, ensure_ascii=False, sort_keys=True
        )
        trace["diagnostic_query_mutated"] = False
        trace["diagnostic_prompt_mutated"] = False
        trace["diagnostic_answer_mutated"] = False
        decision = dict(trace.get("answer_decision") or {})
        if (
            decision.get("commit_mode") == "deterministic_commit"
            and not self.enable_deterministic_answer_commit
        ):
            decision["commit_mode"] = "guided_generation"
            decision["commit_reason"] = "deterministic_commit_disabled"
            trace["answer_decision"] = decision
        return trace

    def _normalize_entity(self, entity: Any) -> str:
        text = str(entity or "").lower()
        text = re.sub(r"\b(the|a|an)\b", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _entity_mentioned(self, entity: Any, text: Any) -> bool:
        normalized_entity = self._normalize_entity(entity)
        normalized_text = self._normalize_entity(text)
        if not normalized_entity or not normalized_text:
            return False
        if normalized_entity in normalized_text:
            return True
        entity_tokens = set(normalized_entity.split())
        text_tokens = set(normalized_text.split())
        return len(entity_tokens) >= 2 and entity_tokens.issubset(text_tokens)

    def _extract_entities_from_claim(self, claim_text: str) -> List[str]:
        text = str(claim_text or "").strip()
        entities = []
        short_candidate = self._claim_to_short_answer_candidate(text)
        if short_candidate:
            entities.append(short_candidate)
        for match in re.finditer(
            r"\b(?:by|from|of|for|with|to|at|in)\s+"
            r"([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})",
            text,
        ):
            entities.append(match.group(1).strip(" ,."))
        subject = re.match(
            r"^([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})\s+",
            text,
        )
        if subject:
            entities.append(subject.group(1).strip(" ,."))
        deduped = []
        seen = set()
        for entity in entities:
            key = self._normalize_entity(entity)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped[:6]

    def _apply_chain_grounding_ready_guard(
        self,
        state: Dict[str, Any],
        docs: List[Dict],
        question: str,
    ) -> bool:
        if not self.enable_relation_grounding:
            chain_trace = self._build_chain_grounding_trace(state, docs, question)
            state["chain_grounding_trace"] = chain_trace
            if (
                state.get("decision") != "ready_to_answer"
                or not chain_trace.get("requires_grounding")
                or chain_trace.get("chain_grounded")
            ):
                return False

            state["ready_guard_triggered"] = True
            fallback_plan = self._build_bridge_fallback_plan(
                question,
                state.get("answer_slots", []),
                state.get("claims", []),
            )
            if fallback_plan and not state.get("guard_extra_round_used"):
                state["retrieval_plan"] = fallback_plan
                state["guard_extra_round_pending"] = True
                state["failure_reason"] = None
                state["decision"] = "continue_retrieval"
            else:
                state["retrieval_plan"] = []
                state["failure_reason"] = state.get("failure_reason") or "chain_ungrounded_final_slot"
                state["decision"] = "force_answer"
            return True

        relation_trace = self._build_relation_grounding_trace(state, docs, question)
        relation_trace["control_action"] = (
            "diagnostic_only" if self.relation_grounding_mode == "diagnostic" else "control"
        )
        chain_trace = relation_trace.get("chain_grounding_trace") or {}
        answer_decision = relation_trace.get("answer_decision") or {}
        state["answer_intent"] = relation_trace.get("answer_intent") or state.get("answer_intent", {})
        state["relation_grounding_trace"] = relation_trace
        state["chain_grounding_trace"] = chain_trace
        state["answer_decision"] = answer_decision

        if state.get("question_type") == "comparison" and self.enable_comparison_executor:
            branch_executable = bool(relation_trace.get("comparison_branch_executable"))
            state["comparison_branch_status"] = relation_trace.get("comparison_branch_status")
            state["comparison_branches"] = relation_trace.get("comparison_branches") or []
            state["comparison_branch_executable"] = branch_executable
            state["comparison_branch_failure_reason"] = relation_trace.get(
                "comparison_branch_failure_reason"
            )
            state["comparison_branch_build_time_ms"] = float(
                relation_trace.get("comparison_branch_build_time_ms") or 0.0
            )
            state["comparison_fallback_to_patch2"] = not (
                self.comparison_branch_mode == "control" and branch_executable
            )
            state["comparison_guidance_applied"] = bool(
                self.comparison_branch_mode == "control" and branch_executable
            )
            state["comparison_view_applied"] = bool(
                self.comparison_branch_mode == "control" and branch_executable
            )
            state["derived_slots_hidden_from_reviewer"] = sum(
                1 for slot in state.get("answer_slots") or []
                if slot.get("slot_role") == "final_answer" and slot.get("derived_output")
            ) if state["comparison_view_applied"] else 0
            if self.comparison_branch_mode == "control" and branch_executable:
                changed = state.get("decision") != "ready_to_answer"
                answer_decision = dict(state.get("answer_decision") or {})
                answer_decision.update({
                    "selected_answer_candidate": None,
                    "candidate_source": "comparison_executor",
                    "commit_mode": "guided_generation",
                    "commit_reason": "comparison_branches_validated",
                    "committed_claim_ids": [],
                })
                state["answer_decision"] = answer_decision
                relation_trace["answer_decision"] = answer_decision
                state["decision"] = "ready_to_answer"
                state["direct_answer_supported"] = True
                state["retrieval_plan"] = []
                state["failure_reason"] = None
                state["guard_extra_round_pending"] = False
                relation_trace["control_action"] = "comparison_branch_guided_generation"
                return changed

        if self.relation_grounding_mode == "diagnostic":
            return False

        if answer_decision.get("commit_mode") == "deterministic_commit":
            changed = state.get("decision") != "ready_to_answer"
            state["decision"] = "ready_to_answer"
            state["direct_answer_supported"] = True
            state["retrieval_plan"] = []
            state["failure_reason"] = None
            state["guard_extra_round_pending"] = False
            return changed

        if (
            state.get("decision") != "ready_to_answer"
            or relation_trace.get("relation_grounded")
        ):
            return False

        state["ready_guard_triggered"] = True
        fallback_plan = self._build_bridge_fallback_plan(
            question,
            state.get("answer_slots", []),
            state.get("claims", []),
        )
        if fallback_plan and not state.get("guard_extra_round_used"):
            state["retrieval_plan"] = fallback_plan
            state["guard_extra_round_pending"] = True
            state["failure_reason"] = None
            state["decision"] = "continue_retrieval"
        else:
            state["guard_extra_round_pending"] = False
            relation_trace["control_action"] = "guided_generation"
            return False
        relation_trace["control_action"] = "focused_retrieval"
        return True

    def _count_new_claims(self, before: Dict[str, Any], after: Dict[str, Any]) -> int:
        before_signatures = {
            (claim.get("doc_id"), claim.get("slot_id"), claim.get("text", "").lower())
            for claim in before.get("claims", [])
        }
        return sum(
            1
            for claim in after.get("claims", [])
            if (claim.get("doc_id"), claim.get("slot_id"), claim.get("text", "").lower()) not in before_signatures
        )

    def _has_valid_retrieval_plan(self, state: Dict[str, Any]) -> bool:
        evidence_slot_ids = self._evidence_slot_ids(state)
        for plan in state.get("retrieval_plan") or []:
            if (
                isinstance(plan, dict)
                and str(plan.get("query") or "").strip()
                and (not evidence_slot_ids or plan.get("slot_id") in evidence_slot_ids)
            ):
                return True
        return False

    def _should_force_no_new_claim(
        self,
        state: Dict[str, Any],
        new_claim_count: int,
        iteration: int,
    ) -> bool:
        return (
            iteration > 0
            and new_claim_count == 0
            and not self._has_valid_retrieval_plan(state)
        )

    def _mark_guard_extra_for_last_chance_retrieval(
        self,
        state: Dict[str, Any],
        new_claim_count: int,
        iteration: int,
    ) -> bool:
        if (
            iteration == self.max_iterations - 1
            and new_claim_count == 0
            and state.get("decision") == "continue_retrieval"
            and self._has_valid_retrieval_plan(state)
            and not state.get("guard_extra_round_used")
        ):
            state["guard_extra_round_pending"] = True
            return True
        return False

    def _rank_docs_by_state(self, docs: List[Dict], state: Dict[str, Any], return_trace: bool = False):
        """Rank docs by slot coverage and claim support, then keep final_max_docs."""
        claims = state.get("claims", [])
        ignored_indices = set(state.get("ignored_docs") or [])
        support_by_doc = {}
        slots_by_doc = {}
        conflict_by_doc = {}

        for claim in claims:
            doc_id = str(claim.get("doc_id", ""))
            slot_id = str(claim.get("slot_id", ""))
            status = claim.get("support_status")
            if status == "support":
                support_by_doc[doc_id] = support_by_doc.get(doc_id, 0) + 1
                slots_by_doc.setdefault(doc_id, set()).add(slot_id)
            elif status == "contradict":
                conflict_by_doc[doc_id] = conflict_by_doc.get(doc_id, 0) + 1

        scored = []
        score_components = {}
        for idx, doc in enumerate(docs):
            doc_id = self._safe_doc_id(doc, idx)
            score = 0
            score += support_by_doc.get(doc_id, 0) * 5
            score += len(slots_by_doc.get(doc_id, set())) * 3
            score -= conflict_by_doc.get(doc_id, 0) * 2
            ignored_penalty = 10 if idx in ignored_indices else 0
            if idx in ignored_indices:
                score -= ignored_penalty
            score_components[doc_id] = {
                "support_claim_count": support_by_doc.get(doc_id, 0),
                "covered_slot_count": len(slots_by_doc.get(doc_id, set())),
                "conflict_claim_count": conflict_by_doc.get(doc_id, 0),
                "ignored_penalty": ignored_penalty,
                "rank_score": score,
            }
            # Keep original order as a weak retriever-score proxy.
            scored.append((score, -idx, doc))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        ranked_docs = [doc for _, _, doc in scored[:self.final_max_docs]]
        if not return_trace:
            return ranked_docs

        before_ids = [self._safe_doc_id(doc, idx) for idx, doc in enumerate(docs)]
        after_ids = [self._safe_doc_id(doc, idx) for idx, doc in enumerate(ranked_docs)]
        trace = {
            "doc_ids_before_rank": before_ids,
            "doc_ids_after_rank": after_ids,
            "doc_titles_after_rank": [self._doc_title(doc) for doc in ranked_docs],
            "ignored_doc_ids": [
                before_ids[idx] for idx in ignored_indices if 0 <= idx < len(before_ids)
            ],
            "support_claim_doc_ids": sorted(support_by_doc.keys()),
            "conflict_claim_doc_ids": sorted(conflict_by_doc.keys()),
            "score_components": {
                doc_id: score_components.get(doc_id, {})
                for doc_id in after_ids
            },
        }
        return ranked_docs, trace

    def _build_adoption_policy(
        self,
        state: Dict[str, Any],
        docs: Optional[List[Dict]] = None,
        failure_reason: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.enable_relation_grounding:
            relation_trace = self._build_relation_grounding_trace(state, docs, question)
            relation_trace["control_action"] = (
                "diagnostic_only" if self.relation_grounding_mode == "diagnostic" else "control"
            )
            chain_trace = relation_trace.get("chain_grounding_trace") or {}
            answer_decision = relation_trace.get("answer_decision") or {}
            state["answer_intent"] = relation_trace.get("answer_intent") or state.get("answer_intent", {})
            state["relation_grounding_trace"] = relation_trace
            state["answer_decision"] = answer_decision
        else:
            relation_trace = {}
            answer_decision = state.get("answer_decision") or {}
            chain_trace = self._build_chain_grounding_trace(state, docs, question)
        comparison_output_contract = build_comparison_output_contract(question or "", state)
        comparison_commit = validate_final_comparison_commit(
            relation_trace,
            comparison_output_contract,
        ) if comparison_output_contract else {
            "eligible": False,
            "candidate": None,
            "committed_claim_ids": [],
            "blocked_reason": "not_comparison",
            "commit_confidence": "blocked",
            "relation_family": "generic",
            "relation_value_type_valid": None,
        }
        comparison_finalization_mode = getattr(
            self,
            "comparison_finalization_mode",
            {
                "diagnostic": "baseline",
                "final_only": "strict_commit",
            }.get(getattr(self, "comparison_branch_mode", "baseline"), "baseline"),
        )
        if comparison_finalization_mode == "strict_commit" and comparison_commit.get("eligible"):
            answer_decision = {
                "selected_answer_candidate": comparison_commit.get("candidate"),
                "candidate_source": "comparison_executor",
                "commit_mode": "deterministic_commit",
                "commit_reason": "strict_comparison_branches_validated",
                "committed_claim_ids": comparison_commit.get("committed_claim_ids") or [],
            }
            relation_trace["answer_decision"] = answer_decision
            relation_trace["control_action"] = "strict_comparison_deterministic_commit"
            state["answer_decision"] = answer_decision
            state["comparison_fallback_to_patch2"] = False
        state["comparison_output_contract"] = comparison_output_contract
        state["chain_grounding_trace"] = chain_trace
        semantic_guidance_applied = bool(
            self.enable_relation_grounding and self.relation_grounding_mode == "control"
        )
        grounded_final_claim_ids = {
            str(claim_id)
            for claim_id in chain_trace.get("grounded_final_claim_ids") or []
        }
        claim_grounding = chain_trace.get("claim_grounding") or {}
        semantic_bindings = {
            str(binding.get("claim_id")): binding
            for binding in relation_trace.get("claim_bindings") or []
            if binding.get("claim_id") and binding.get("source") != "reviewer_answer_candidate"
        }
        resolved_intermediate_entities = [
            entity.get("entity")
            for entity in chain_trace.get("resolved_intermediate_entities") or []
            if entity.get("entity")
        ]
        trusted_claims = [
            claim for claim in state.get("claims", [])
            if claim.get("support_status") == "support"
        ]
        validated_claim_ids = {
            claim_id
            for claim_id, binding in semantic_bindings.items()
            if binding.get("eligible")
        }
        validated_claims = [
            claim for claim in trusted_claims
            if str(claim.get("claim_id")) in validated_claim_ids
        ]
        unvalidated_claims = [
            claim for claim in trusted_claims
            if str(claim.get("claim_id")) not in validated_claim_ids
        ]
        conflicted_claims = [
            claim for claim in state.get("claims", [])
            if claim.get("support_status") == "contradict"
        ]
        evidence_slot_ids = self._evidence_slot_ids(state)
        insufficient_slots = [
            slot["slot_id"] for slot in state.get("answer_slots", [])
            if slot.get("slot_id") in evidence_slot_ids
            and slot.get("status") in {"missing", "partial", "conflicted"}
        ]
        must_use_slots = [
            slot["slot_id"] for slot in state.get("answer_slots", [])
            if slot.get("slot_id") in evidence_slot_ids
            and slot.get("status") == "supported"
        ]
        final_answer_slot_ids = self._final_answer_slot_ids(state.get("answer_slots", []))
        if not self.enable_relation_grounding and not final_answer_slot_ids and state.get("question_type") == "comparison":
            final_answer_slot_ids = {
                slot.get("slot_id")
                for slot in state.get("answer_slots", [])
                if slot.get("status") == "supported"
            }
        slots_by_id = {slot.get("slot_id"): slot for slot in state.get("answer_slots", [])}
        claims_by_id = {
            str(claim.get("claim_id")): claim
            for claim in state.get("claims", [])
            if claim.get("claim_id") is not None
        }
        answer_candidates = []
        candidate_records = []
        comparison_guidance_applied = bool(
            getattr(self, "comparison_branch_mode", "diagnostic") == "control"
            and relation_trace.get("comparison_branch_executable")
        )
        comparison_candidate = str(
            relation_trace.get("counterfactual_answer_candidate") or ""
        ).strip()
        if comparison_guidance_applied and comparison_candidate:
            answer_candidates.append(comparison_candidate)
            candidate_records.append({
                "candidate": comparison_candidate,
                "source": "comparison_executor",
                "slot_id": None,
                "claim_id": None,
                "claim_text": "",
                "slot_role": "derived_output",
                "has_claim_provenance": bool(
                    relation_trace.get("counterfactual_claim_ids")
                ),
                "chain_grounded": True,
                "matches_resolved_intermediate": False,
                "grounding_reason": "comparison_branches_validated",
                "target_relation_match": True,
                "candidate_role_valid": True,
                "candidate_answer_type_match": True,
                "eligible_for_commit": False,
            })
        answer_candidate = str(state.get("answer_candidate") or "").strip()
        if answer_candidate:
            answer_candidate_claim_ids = [
                str(claim_id)
                for claim_id in state.get("answer_candidate_claim_ids") or []
                if claim_id is not None
            ]
            candidate_claims = [
                claims_by_id.get(claim_id)
                for claim_id in answer_candidate_claim_ids
                if claims_by_id.get(claim_id)
            ]
            candidate_is_final = any(
                claim.get("slot_id") in final_answer_slot_ids
                for claim in candidate_claims
            )
            candidate_is_intermediate = bool(candidate_claims) and not candidate_is_final
            candidate_slot_id = candidate_claims[0].get("slot_id") if candidate_claims else None
            candidate_slot = slots_by_id.get(candidate_slot_id) or {}
            candidate_binding_valid = semantic_guidance_applied and any(binding.get("eligible") for binding in (
                semantic_bindings.get(claim_id) or {} for claim_id in answer_candidate_claim_ids
            ))
            if candidate_binding_valid:
                answer_candidates.append(answer_candidate)
            candidate_records.append({
                "candidate": answer_candidate,
                "source": "reviewer_answer_candidate",
                "slot_id": candidate_slot_id,
                "claim_id": answer_candidate_claim_ids[0] if answer_candidate_claim_ids else None,
                "claim_text": " ".join(
                    str(claim.get("text") or "")
                    for claim in candidate_claims[:3]
                    if claim
                ),
                "slot_role": (
                    "final_answer"
                    if candidate_is_final
                    else "intermediate_entity" if candidate_is_intermediate else candidate_slot.get("slot_role", "unknown")
                ),
                "answer_candidate_claim_ids": answer_candidate_claim_ids,
                "has_claim_provenance": bool(answer_candidate_claim_ids),
                "chain_grounded": bool(grounded_final_claim_ids.intersection(answer_candidate_claim_ids)),
                "matches_resolved_intermediate": any(
                    self._normalize_entity(answer_candidate) == self._normalize_entity(entity)
                    for entity in resolved_intermediate_entities
                ),
                "grounding_reason": "matched_grounded_final_claim" if grounded_final_claim_ids.intersection(answer_candidate_claim_ids) else chain_trace.get("ungrounded_reason"),
            })
        for claim in trusted_claims:
            if claim.get("slot_id") not in final_answer_slot_ids:
                continue
            binding = semantic_bindings.get(str(claim.get("claim_id"))) or {}
            candidate = str(binding.get("candidate") or claim.get("object") or "").strip()
            if not candidate:
                candidate = self._claim_to_answer_candidate(
                    claim.get("text", ""),
                    state.get("answer_type"),
                    question,
                )
            if (
                semantic_guidance_applied
                and candidate
                and binding.get("eligible")
                and candidate not in answer_candidates
            ):
                answer_candidates.append(candidate)
            if candidate:
                slot = slots_by_id.get(claim.get("slot_id")) or {}
                claim_id = str(claim.get("claim_id"))
                grounding = claim_grounding.get(claim_id) or {}
                candidate_records.append({
                    "candidate": candidate,
                    "source": "validated_claim" if binding.get("eligible") else "unvalidated_claim",
                    "slot_id": claim.get("slot_id"),
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("text", ""),
                    "slot_role": slot.get("slot_role"),
                    "has_claim_provenance": bool(claim.get("claim_id")),
                    "chain_grounded": claim_id in grounded_final_claim_ids,
                    "matches_resolved_intermediate": any(
                        self._normalize_entity(candidate) == self._normalize_entity(entity)
                        for entity in resolved_intermediate_entities
                    ),
                    "grounding_reason": grounding.get("reason") or chain_trace.get("ungrounded_reason"),
                    "target_relation_match": bool(binding.get("target_relation_match")),
                    "candidate_role_valid": bool(binding.get("candidate_role_valid")),
                    "candidate_answer_type_match": bool(binding.get("candidate_answer_type_match")),
                    "eligible_for_commit": bool(binding.get("eligible")),
                    "candidate_is_question_entity": bool(binding.get("candidate_is_question_entity")),
                    "candidate_is_intermediate_entity": bool(binding.get("candidate_is_intermediate_entity")),
                    "binding_direction": binding.get("binding_direction"),
                    "anchor_entity": binding.get("anchor_entity"),
                    "anchor_endpoint": binding.get("anchor_endpoint"),
                    "candidate_endpoint": binding.get("candidate_endpoint"),
                    "entity_binding_valid": bool(binding.get("entity_binding_valid")),
                    "provenance_valid": bool(binding.get("provenance_valid")),
                })
        committed_candidate = str(answer_decision.get("selected_answer_candidate") or "").strip()
        if committed_candidate and committed_candidate not in answer_candidates:
            answer_candidates.insert(0, committed_candidate)
            candidate_records.insert(0, {
                "candidate": committed_candidate,
                "source": answer_decision.get("candidate_source") or "comparison_executor",
                "slot_id": None,
                "claim_id": (answer_decision.get("committed_claim_ids") or [None])[0],
                "claim_text": "",
                "slot_role": "final_answer",
                "has_claim_provenance": bool(answer_decision.get("committed_claim_ids")),
                "chain_grounded": True,
                "matches_resolved_intermediate": False,
                "grounding_reason": answer_decision.get("commit_reason"),
                "target_relation_match": True,
                "candidate_role_valid": True,
                "candidate_answer_type_match": True,
                "eligible_for_commit": True,
            })
        counterfactual_candidate = str(relation_trace.get("counterfactual_answer_candidate") or "").strip()
        if counterfactual_candidate and not committed_candidate:
            candidate_records.append({
                "candidate": counterfactual_candidate,
                "source": relation_trace.get("counterfactual_candidate_source") or "comparison_executor",
                "slot_id": None,
                "claim_id": (relation_trace.get("counterfactual_claim_ids") or [None])[0],
                "claim_text": "",
                "slot_role": "final_answer",
                "has_claim_provenance": bool(relation_trace.get("counterfactual_claim_ids")),
                "chain_grounded": bool(relation_trace.get("relation_grounded")),
                "matches_resolved_intermediate": False,
                "grounding_reason": "counterfactual_only",
                "target_relation_match": bool(relation_trace.get("target_relation_match")),
                "candidate_role_valid": True,
                "candidate_answer_type_match": True,
                "eligible_for_commit": False,
                "counterfactual_only": True,
            })
        policy_trace = self._score_answer_candidates(
            candidate_records,
            answer_candidates,
            state.get("answer_type"),
            state.get("question_type"),
            question,
        )
        selected_answer_candidate = committed_candidate or self._select_answer_candidate(
            answer_candidates,
            state.get("answer_type"),
            state.get("question_type"),
            question,
            candidate_records,
        )
        compressed_answer_candidate = self._compress_short_answer(answer_candidate) if answer_candidate else None
        candidate_source = None
        if selected_answer_candidate:
            candidate_source = (
                "reviewer_answer_candidate"
                if selected_answer_candidate in {answer_candidate, compressed_answer_candidate}
                else "unvalidated_claim"
            )
            for record in policy_trace.get("ranked_candidates") or []:
                if record.get("candidate") == selected_answer_candidate:
                    candidate_source = record.get("source") or candidate_source
                    break
        return {
            "trusted_claims": trusted_claims,
            "validated_claims": validated_claims,
            "unvalidated_claims": unvalidated_claims,
            "answer_candidates": answer_candidates[:8],
            "selected_answer_candidate": selected_answer_candidate,
            "candidate_source": candidate_source,
            "answer_policy_trace": policy_trace,
            "answer_intent": state.get("answer_intent", {}),
            "relation_grounding_trace": relation_trace,
            "answer_decision": answer_decision,
            "semantic_guidance_applied": semantic_guidance_applied,
            "comparison_guidance_applied": comparison_guidance_applied,
            "comparison_output_contract": comparison_output_contract,
            "comparison_output_contract_available": bool(comparison_output_contract),
            "comparison_output_contract_applied": False,
            "comparison_finalization_mode": comparison_finalization_mode,
            "comparison_commit_eligible": bool(comparison_commit.get("eligible")),
            "comparison_commit_blocked_reason": comparison_commit.get("blocked_reason"),
            "comparison_commit_confidence": comparison_commit.get("commit_confidence"),
            "comparison_commit_relation_family": comparison_commit.get("relation_family"),
            "comparison_commit_relation_value_type_valid": comparison_commit.get(
                "relation_value_type_valid"
            ),
            "comparison_slot_associated_rejected": (
                comparison_commit.get("blocked_reason") == "slot_associated_path_rejected"
            ),
            "comparison_final_candidate_matches_option": bool(
                comparison_commit.get("eligible") and comparison_commit.get("candidate")
            ),
            "comparison_guided_generation": False,
            "comparison_final_commit_saved_generation": bool(
                comparison_output_contract
                and answer_decision.get("commit_mode") == "deterministic_commit"
            ),
            "comparison_branches": relation_trace.get("comparison_branches") or [],
            "comparison_branch_status": relation_trace.get("comparison_branch_status"),
            "comparison_branch_executable": bool(
                relation_trace.get("comparison_branch_executable")
            ),
            "comparison_branch_failure_reason": relation_trace.get(
                "comparison_branch_failure_reason"
            ),
            "comparison_fallback_to_patch2": bool(
                state.get("question_type") == "comparison"
                and not comparison_guidance_applied
                and not (
                    comparison_finalization_mode == "strict_commit"
                    and comparison_commit.get("eligible")
                )
            ),
            "diagnostic_candidate_leakage": bool(
                not semantic_guidance_applied
                and answer_candidates
                and answer_decision.get("commit_mode") != "deterministic_commit"
            ),
            "counterfactual_answer_candidate": relation_trace.get("counterfactual_answer_candidate"),
            "counterfactual_candidate_source": relation_trace.get("counterfactual_candidate_source"),
            "would_commit_under_v2_7": bool(relation_trace.get("would_commit_under_v2_7")),
            "generated_without_validated_candidate": bool(
                answer_decision.get("commit_mode") in {"guided_generation", "forced_generation"}
                and not validated_claims
            ),
            "chain_grounding_trace": chain_trace,
            "chain_coverage": chain_trace.get("chain_coverage"),
            "answer_type": state.get("answer_type"),
            "ignored_docs": state.get("ignored_docs", []),
            "conflict_resolution": [
                {
                    "slot_id": claim.get("slot_id"),
                    "claim_id": claim.get("claim_id"),
                    "resolution": "Do not rely on this contradicted or isolated claim unless no supported claim exists.",
                }
                for claim in conflicted_claims
            ],
            "must_use_slots": must_use_slots,
            "insufficient_slots": insufficient_slots,
            "answer_style": "short_answer",
            "fallback_instruction": (
                "If trusted_claims are sparse or insufficient_slots remain, answer with the most likely "
                "short answer supported by the provided documents. Do not output only uncertainty for "
                "closed-form QA unless no plausible answer candidate exists."
            ),
            "failure_reason": failure_reason or state.get("failure_reason"),
        }

    def _render_adoption_policy(self, policy: Dict[str, Any]) -> str:
        """Render internal policy as natural-language guidance for the generator."""
        trusted_claims = policy.get("trusted_claims") or []
        validated_claims = policy.get("validated_claims") or []
        answer_candidates = policy.get("answer_candidates") or []
        insufficient_slots = policy.get("insufficient_slots") or []
        ignored_docs = policy.get("ignored_docs") or []
        conflict_resolution = policy.get("conflict_resolution") or []
        chain_trace = policy.get("chain_grounding_trace") or {}
        grounded_claim_ids = {
            str(claim_id)
            for claim_id in chain_trace.get("grounded_final_claim_ids") or []
        }

        lines = []
        output_contract = policy.get("comparison_output_contract") or {}
        if output_contract and policy.get("comparison_output_contract_applied"):
            options = [str(option) for option in output_contract.get("option_entities") or []]
            lines.append("Comparison output contract:")
            if output_contract.get("output_role") == "boolean":
                lines.append("- Return exactly yes or no after comparing the supported operands.")
            elif options:
                lines.append("- Return exactly one original question option: " + " | ".join(options[:4]))
                lines.append("- Do not return the compared date, number, attribute, or intermediate entity.")
            else:
                lines.append("- Return the compared entity selected by the question, not an operand value.")
        if policy.get("comparison_guidance_applied"):
            lines.append("Validated comparison result:")
            selected_answer_candidate = str(policy.get("selected_answer_candidate") or "").strip()
            if selected_answer_candidate:
                lines.append(f"- Return the original question option: {selected_answer_candidate}")
            for branch in policy.get("comparison_branches") or []:
                lines.append(
                    f"- {branch.get('option')}: {branch.get('raw_value')}"
                    + (f" via {branch.get('intermediate_entity')}" if branch.get("intermediate_entity") else "")
                )
        if policy.get("semantic_guidance_applied"):
            lines.append("Relation-validated answer facts:")
            if validated_claims:
                for claim in validated_claims[:12]:
                    text = str(claim.get("text") or "").strip()
                    doc_id = claim.get("doc_id")
                    if text:
                        lines.append(f"- {text}" + (f" (source document id: {doc_id})" if doc_id else ""))
            else:
                lines.append("- No final-answer claim passed relation validation; infer cautiously from the most relevant document text.")

            if grounded_claim_ids:
                lines.append("Grounded final-answer facts:")
                for claim in validated_claims[:12]:
                    if str(claim.get("claim_id")) not in grounded_claim_ids:
                        continue
                    text = str(claim.get("text") or "").strip()
                    if text:
                        lines.append(f"- {text}")
            elif chain_trace and chain_trace.get("requires_grounding"):
                lines.append("Grounding constraint:")
                lines.append("- Prefer final-answer facts bound to resolved intermediate entities; avoid outputting an intermediate entity as the answer.")

        if answer_candidates:
            lines.append("Preferred short answer candidates:")
            selected_answer_candidate = str(policy.get("selected_answer_candidate") or "").strip()
            if selected_answer_candidate:
                lines.append(f"- {selected_answer_candidate}")
            for candidate in answer_candidates[:8]:
                if candidate == selected_answer_candidate:
                    continue
                lines.append(f"- {candidate}")
            lines.append("- Prefer the shortest candidate that fully answers the question.")

        if ignored_docs:
            lines.append("Documents to avoid:")
            lines.append("- Avoid document indices: " + ", ".join(str(idx) for idx in ignored_docs[:12]))

        if conflict_resolution:
            lines.append("Conflict handling:")
            lines.append("- Prefer directly supported facts over contradicted or isolated claims.")

        if insufficient_slots:
            lines.append("Missing or partial evidence:")
            lines.append("- Some required evidence is incomplete. Still provide the best supported short answer when a plausible candidate appears in the documents.")

        lines.append("Output requirements:")
        lines.append("- Return only the answer text, not explanations or internal policy labels.")
        lines.append("- Never output words like trusted_claims, insufficient_slots, must_use_slots, ignored_docs, or conflict_resolution.")
        return "\n".join(lines)

    def _should_short_circuit_single_hop(self, state: Dict[str, Any], coverage: Dict[str, Any]) -> bool:
        return (
            state.get("question_type") == "single_hop"
            and bool(state.get("direct_answer_supported"))
            and coverage.get("total", 0) > 0
            and coverage.get("coverage", 0.0) >= 1.0
            and coverage.get("conflicted", 0) == 0
        )

    def _fallback_answer_from_policy(self, policy: Optional[Dict[str, Any]]) -> Optional[str]:
        if not policy:
            return None
        selected = str(policy.get("selected_answer_candidate") or "").strip()
        if selected:
            return selected
        answer_candidates = [
            str(candidate).strip()
            for candidate in policy.get("answer_candidates") or []
            if str(candidate).strip()
        ]
        if answer_candidates:
            selected = self._select_answer_candidate(answer_candidates, policy.get("answer_type"))
            if selected:
                return selected
            return answer_candidates[0]
        if not policy.get("semantic_guidance_applied"):
            return None
        candidates = []
        for claim in policy.get("trusted_claims") or []:
            text = str(claim.get("text") or "").strip()
            if text:
                short_candidate = self._claim_to_short_answer_candidate(text)
                if short_candidate:
                    candidates.append(short_candidate)
                candidates.append(text)
        if not candidates:
            return None
        candidates.sort(key=lambda text: (len(text.split()), len(text)))
        return candidates[0]

    def _select_answer_candidate(
        self,
        candidates: List[str],
        answer_type: Optional[str],
        question_type: Optional[str] = None,
        question: Optional[str] = None,
        candidate_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        normalized = []
        for candidate in candidates or []:
            candidate = self._compress_short_answer(candidate) or str(candidate).strip()
            if not candidate or candidate in normalized:
                continue
            normalized.append(candidate)

        trace = self._score_answer_candidates(
            candidate_records or [],
            normalized,
            answer_type,
            question_type,
            question,
        )
        ranked = trace.get("ranked_candidates") or []
        if ranked:
            return ranked[0].get("candidate")

        matching = [
            candidate for candidate in normalized
            if self._candidate_matches_answer_type(candidate, answer_type)
        ]
        pool = matching or normalized
        if not pool:
            return None
        pool.sort(key=lambda text: (len(text.split()), len(text)))
        return pool[0]

    def _score_answer_candidates(
        self,
        candidate_records: List[Dict[str, Any]],
        candidates: List[str],
        answer_type: Optional[str],
        question_type: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        seen = set()
        records = []
        records_by_candidate = {
            str(record.get("candidate") or "").strip(): record
            for record in candidate_records or []
            if str(record.get("candidate") or "").strip()
        }
        for candidate in candidates or []:
            candidate = self._compress_short_answer(candidate) or str(candidate).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            record = dict(records_by_candidate.get(candidate) or {})
            record.setdefault("candidate", candidate)
            record.setdefault("source", "unknown")
            records.append(record)

        q = str(question or "").lower()
        answer_type = str(answer_type or "").lower()
        question_type = str(question_type or "").lower()
        ranked = []
        filtered = []
        for record in records:
            candidate = str(record.get("candidate") or "").strip()
            if not candidate:
                continue
            reasons = []
            score = 0
            if self._candidate_matches_answer_type(candidate, answer_type):
                score += 25
                reasons.append("answer_type_match")
            else:
                score -= 15
                reasons.append("answer_type_mismatch")
            if record.get("slot_role") in {"final_answer", "direct_answer"}:
                score += 20
                reasons.append("final_slot")
            if record.get("source") == "reviewer_answer_candidate":
                if record.get("has_claim_provenance"):
                    score += 8
                    reasons.append("reviewer_answer_candidate_with_provenance")
                else:
                    score -= 20
                    reasons.append("missing_candidate_claim_provenance")
            if record.get("source") == "validated_claim":
                score += 24
                reasons.append("relation_validated_claim")
            elif record.get("source") == "unvalidated_claim":
                score -= 32
                reasons.append("unvalidated_claim_penalty")
            if record.get("chain_grounded"):
                score += 30
                reasons.append("chain_grounded")
            elif question_type in {"bridge", "multihop"} or record.get("slot_role") == "intermediate_entity":
                score -= 28
                reasons.append("ungrounded_or_intermediate_candidate")
            if question_type in {"bridge", "multihop"} and record.get("matches_resolved_intermediate"):
                score -= 45
                reasons.append("resolved_intermediate_entity_penalty")
            overlap = self._question_candidate_overlap(candidate, q)
            if overlap and question_type == "comparison":
                score += min(overlap * 3, 18)
                reasons.append("comparison_option_overlap")
            elif record.get("candidate_is_question_entity"):
                score -= 40
                reasons.append("question_entity_penalty")
            if record.get("candidate_is_intermediate_entity"):
                score -= 45
                reasons.append("intermediate_entity_penalty")
            if answer_type in {"number", "date"} and re.search(r"\([^)]*\d[^)]*\)", str(record.get("claim_text") or "")):
                score += 12
                reasons.append("parenthetical_specificity")
            if answer_type == "number" and re.search(r"\b(?:seat|seated|capacity|people)\b", q):
                if re.search(r"\b(?:seated|capacity)\b", candidate, re.IGNORECASE):
                    score += 16
                    reasons.append("number_context_match")
                elif re.fullmatch(r"[\d,]+", candidate):
                    score -= 8
                    reasons.append("bare_number_penalty")
            if question_type == "comparison" or re.search(r"\b(which|who)\b.*\b(first|older|younger|larger|smaller|more|less)\b", q):
                if answer_type in {"title_or_entity", "person", "location", "phrase"} and re.fullmatch(r"\d{4}(?:[-/]\d{2}[-/]\d{2})?", candidate):
                    score -= 60
                    reasons.append("comparison_intermediate_value_penalty")
                if candidate.lower() in q:
                    score += 22
                    reasons.append("question_option_candidate")
            score -= min(len(candidate.split()), 12)
            ranked.append({
                "candidate": candidate,
                "score": score,
                "source": record.get("source"),
                "slot_id": record.get("slot_id"),
                "claim_id": record.get("claim_id"),
                "chain_grounded": bool(record.get("chain_grounded")),
                "has_claim_provenance": bool(record.get("has_claim_provenance")),
                "matches_resolved_intermediate": bool(record.get("matches_resolved_intermediate")),
                "grounding_reason": record.get("grounding_reason"),
                "target_relation_match": bool(record.get("target_relation_match")),
                "candidate_role_valid": bool(record.get("candidate_role_valid")),
                "candidate_answer_type_match": bool(record.get("candidate_answer_type_match")),
                "eligible_for_commit": bool(record.get("eligible_for_commit")),
                "reasons": reasons,
            })
        ranked.sort(key=lambda item: (-item["score"], len(str(item["candidate"]).split()), len(str(item["candidate"]))))
        return {
            "ranked_candidates": ranked[:8],
            "filtered_candidates": filtered,
            "selection_reason": "highest_scored_candidate" if ranked else "no_candidate",
        }

    def _question_candidate_overlap(self, candidate: str, question_lower: str) -> int:
        stopwords = {
            "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "was", "were",
            "is", "are", "what", "which", "who", "whom", "where", "when", "how", "many",
            "much", "that", "both", "same", "first", "older", "younger", "larger", "smaller",
        }
        q_tokens = {
            token for token in re.findall(r"[a-z0-9]+", question_lower)
            if len(token) > 2 and token not in stopwords
        }
        c_tokens = {
            token for token in re.findall(r"[a-z0-9]+", str(candidate or "").lower())
            if len(token) > 2 and token not in stopwords
        }
        return len(q_tokens.intersection(c_tokens))

    def _candidate_matches_answer_type(self, candidate: str, answer_type: Optional[str]) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        answer_type = str(answer_type or "").lower()
        if answer_type == "yes_no":
            return text.lower() in {"yes", "no"}
        if answer_type == "number":
            return bool(re.search(r"\d", text))
        if answer_type == "date":
            return bool(re.search(r"\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.IGNORECASE))
        if answer_type in {"person", "location", "title_or_entity"}:
            return bool(re.search(r"\b[A-Z][A-Za-z0-9$'.-]+", text)) and len(text.split()) <= 8
        return len(text.split()) <= 12

    def _claim_to_answer_candidate(
        self,
        claim_text: str,
        answer_type: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Optional[str]:
        text = str(claim_text or "").strip()
        if not text:
            return None
        answer_type = str(answer_type or "").lower()
        question_lower = str(question or "").lower()

        if answer_type == "location" or question_lower.startswith("where"):
            location_patterns = [
                r"\b(?:born|died|located|based|headquartered|situated)\s+(?:in|at|near)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})",
                r"\b(?:city|country|state|province|region|location)\s+(?:is|was)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})",
                r"\b(?:in|at|from)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})(?:[.,]|$)",
            ]
            for pattern in location_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip(" ,.")

        if answer_type == "date":
            date_match = re.search(
                r"\b(?:in|on|during)\s+([A-Z][a-z]+(?:\s+\d{1,2})?,?\s+\d{4}|\d{4})\b",
                text,
            )
            if date_match:
                return date_match.group(1).strip(" ,.")

        return self._claim_to_short_answer_candidate(text)

    def _claim_to_short_answer_candidate(self, claim_text: str) -> Optional[str]:
        text = str(claim_text or "").strip()
        if not text:
            return None

        quoted = re.findall(r'"([^"\n]{2,80})"', text)
        if quoted:
            quoted.sort(key=lambda item: (len(item.split()), len(item)))
            return quoted[0].strip()

        parenthetical_number = re.search(r"\(([^)]*\d[^)]*)\)", text)
        if parenthetical_number:
            return parenthetical_number.group(1).strip(" ,.")

        office_match = re.search(
            r"\b(?:served|serves|held|holds|appointed|was|is|became)\s+as\s+"
            r"(?:the\s+|a\s+|an\s+)?(?:first\s+female\s+)?"
            r"([A-Z][A-Za-z]+(?:\s+(?:of|for|and|the|[A-Z][A-Za-z]+)){0,7})",
            text,
        )
        if office_match:
            candidate = office_match.group(1).strip(" ,.")
            candidate = re.sub(r"\s+of the United States$", "", candidate).strip(" ,.")
            if candidate:
                return candidate

        date_match = re.search(
            r"\b(?:in|on|during)\s+([A-Z][a-z]+(?:\s+\d{1,2})?,?\s+\d{4}|\d{4})\b",
            text,
        )
        if date_match:
            return date_match.group(1).strip(" ,.")

        by_match = re.search(r"\b(?:by|created by|written by|performed by)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,4})", text)
        if by_match:
            return by_match.group(1).strip(" ,.")

        role_match = re.search(r"\b(?:rapper|artist|singer|actor|actress|writer|composer|director|player)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,3})", text)
        if role_match:
            return role_match.group(1).strip(" ,.")

        subject_match = re.match(r"^([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})\s+(?:is|was|were|won|created|wrote|served|appeared|features?|portrayed|directed|played|held|starred)\b", text)
        if subject_match:
            return subject_match.group(1).strip(" ,.")

        return None

    def _looks_like_explanation(self, text: str) -> bool:
        lower = str(text or "").lower()
        explanation_markers = (
            "provided documents",
            "based on",
            "therefore",
            "however",
            "while ",
            "there is no",
            "do not specify",
            "does not specify",
            "cannot be determined",
            "unknown",
            "the most supported answer",
            "the answer is",
            "answer is",
            "according to",
        )
        return any(marker in lower for marker in explanation_markers)

    def _extract_short_answer_candidate(self, answer: str) -> Optional[str]:
        text = str(answer or "").strip()
        if not text:
            return None

        marker_patterns = [
            r"(?:the\s+)?most\s+supported\s+answer\s+is\s*:?\s*(.+)$",
            r"(?:the\s+)?answer\s+is\s*:?\s*(.+)$",
            r"(?:final\s+answer|answer)\s*:\s*(.+)$",
        ]
        for pattern in marker_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                candidate = self._compress_short_answer(match.group(1))
                if candidate and not self._is_uncertainty_answer(candidate):
                    return candidate

        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if len(re.split(r"(?<=[.!?])\s+", line)) > 1:
                continue
            candidate = self._compress_short_answer(line)
            if candidate and not self._looks_like_explanation(candidate) and not self._is_uncertainty_answer(candidate):
                return candidate

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        for sentence in sentences:
            candidate = self._compress_short_answer(sentence)
            if candidate and not self._looks_like_explanation(candidate) and not self._is_uncertainty_answer(candidate):
                return candidate

        return None

    def _compress_short_answer(self, answer: str) -> Optional[str]:
        candidate = str(answer or "").strip()
        if not candidate:
            return None

        candidate = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", candidate).strip()
        candidate = re.sub(
            r"^(?:therefore|thus|so|based on (?:the )?(?:documents|context|evidence),?|according to (?:the )?(?:documents|context|evidence),?)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(
            r"^(?:the\s+)?(?:answer|final answer|most supported answer)\s+(?:is|would be)\s*:?\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        if not candidate:
            return None

        quoted = re.findall(r"'([^'\n]{2,80})'|\"([^\"\n]{2,80})\"", candidate)
        quoted = [first or second for first, second in quoted]
        if quoted:
            quoted.sort(key=lambda item: (len(item.split()), len(item)))
            return quoted[0].strip()

        candidate = candidate.strip(" \"'`.,;:")

        candidate = re.split(r"\n{2,}", candidate, maxsplit=1)[0].strip()
        candidate = re.split(r"\s+(?:because|since|as shown by|according to)\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        candidate = candidate.strip(" \"'`.,;:")

        numeric = re.fullmatch(r"(\d+(?:\.\d+)?)\s+(?:episodes?|years?|people|points|votes|seasons?|games?|months?|days?)", candidate, flags=re.IGNORECASE)
        if numeric:
            return numeric.group(1)

        predicate_object = re.search(
            r"\b(?:presented|played|ended up|located|filmed|served|appeared)\s+(?:in|at|with|by|on)\s+([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,6}|[a-z]+)\b",
            candidate,
        )
        if predicate_object:
            return predicate_object.group(1).strip(" ,.")

        final_value = re.search(r"\bis\s+([A-Za-z]+|\d+)\.?$", candidate, flags=re.IGNORECASE)
        if final_value:
            return final_value.group(1).strip(" ,.")

        subject_winner = re.match(
            r"^(?:The\s+)?([A-Z][A-Za-z0-9$'.-]+(?:\s+[A-Z][A-Za-z0-9$'.-]+){0,5})\s+(?:won|defeated|claimed)\b",
            candidate,
        )
        if subject_winner:
            return subject_winner.group(1).strip(" ,.")

        title_tail = re.match(
            r"^(?:the\s+)?(?:\d{4}\s+)?(?:[A-Za-z/-]+\s+){1,6}(?:film|movie|novel|song|album|series|classic)\s+(.+)$",
            candidate,
            flags=re.IGNORECASE,
        )
        if title_tail:
            tail = title_tail.group(1).strip(" \"'`.,;:")
            if 0 < len(tail.split()) <= 8:
                return tail

        if len(candidate.split()) <= 12:
            return candidate
        return None

    def _is_uncertainty_answer(self, answer: str) -> bool:
        normalized = re.sub(r"[\s`\"':,.\[\]{}-]+", "_", str(answer or "").lower()).strip("_")
        return normalized in {
            "unknown",
            "not_available",
            "not_specified",
            "cannot_determine",
            "insufficient_evidence",
            "none",
            "no_answer",
        }

    def _should_aggressively_clean_answer(
        self,
        policy: Optional[Dict[str, Any]],
        evidence_state: Optional[Dict[str, Any]],
    ) -> bool:
        dataset_name = str(self.config["dataset_name"]).lower()
        question_type = str((evidence_state or {}).get("question_type", "")).lower()
        return (
            dataset_name in {"triviaqa", "ambig_qa", "ambigqa"}
            or question_type == "single_hop"
        )

    def _clean_final_answer(
        self,
        pred: str,
        policy: Optional[Dict[str, Any]] = None,
        evidence_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        answer = str(pred or "").strip()
        if not answer:
            return self._fallback_answer_from_policy(policy) or answer

        markers = ["final answer:", "answer:"]
        lower = answer.lower()
        for marker in markers:
            if marker in lower:
                answer = answer[lower.rfind(marker) + len(marker):].strip()
                lower = answer.lower()

        policy_terms = {
            "trusted_claims",
            "insufficient_slots",
            "must_use_slots",
            "ignored_docs",
            "conflict_resolution",
            "fallback_instruction",
        }
        normalized = re.sub(r"[\s`\"':,.\[\]{}]+", "_", lower).strip("_")
        has_policy_leak = any(term in lower or term == normalized for term in policy_terms)

        uncertainty_only = self._is_uncertainty_answer(answer)
        if has_policy_leak or uncertainty_only:
            fallback = self._fallback_answer_from_policy(policy)
            if fallback:
                return fallback

        aggressive_clean = self._should_aggressively_clean_answer(policy, evidence_state)
        if aggressive_clean:
            short_candidate = self._extract_short_answer_candidate(answer)
            if short_candidate:
                return short_candidate

            if self._looks_like_explanation(answer):
                fallback = self._fallback_answer_from_policy(policy)
                if fallback and len(fallback.split()) <= 12:
                    return fallback

        first_line = next((line.strip() for line in answer.splitlines() if line.strip()), answer)
        if len(answer.split()) > 80 and len(first_line.split()) <= 25:
            answer = first_line
        return answer.strip()

    def _committed_answer_from_policy(self, policy: Optional[Dict[str, Any]]) -> Optional[str]:
        if not policy:
            return None
        decision = policy.get("answer_decision") or {}
        if decision.get("commit_mode") != "deterministic_commit":
            return None
        comparison_strict_commit = (
            policy.get("comparison_finalization_mode") == "strict_commit"
            and policy.get("comparison_commit_eligible")
        )
        if not self.enable_deterministic_answer_commit and not comparison_strict_commit:
            return None
        candidate = str(decision.get("selected_answer_candidate") or "").strip()
        return candidate or None

    def _finalize_answer(
        self,
        generated_answer: Optional[str],
        policy: Optional[Dict[str, Any]],
        evidence_state: Optional[Dict[str, Any]],
    ) -> str:
        committed = self._committed_answer_from_policy(policy)
        if committed:
            return committed
        return self._clean_final_answer(generated_answer or "", policy, evidence_state)

    def _generate_final_answers(
        self,
        prompts: List[str],
        policies: List[Optional[Dict[str, Any]]],
        evidence_states: List[Dict[str, Any]],
    ) -> List[str]:
        answers: List[Optional[str]] = [None] * len(prompts)
        generation_indices = []
        generation_prompts = []
        for idx, prompt in enumerate(prompts):
            committed = self._committed_answer_from_policy(policies[idx])
            if committed:
                answers[idx] = committed
            else:
                generation_indices.append(idx)
                generation_prompts.append(prompt)

        generated = self._batched_generate(self.generator, generation_prompts) if generation_prompts else []
        for offset, idx in enumerate(generation_indices):
            answers[idx] = self._finalize_answer(
                generated[offset],
                policies[idx],
                evidence_states[idx],
            )
        return [str(answer or "") for answer in answers]

    def parse_reviewer_response(self, response: str) -> Dict[str, Any]:
        """Backward-compatible parser that does not default parse failures to ready."""
        parsed, error = self._parse_json_response(response)
        if parsed is None:
            return {
                "decision": "review_failed",
                "evaluation": {"sufficiency": "unknown", "conflicts": "unknown", "credibility": "unknown"},
                "explanation": f"Failed to parse reviewer response: {error}",
                "new_query": None,
                "docs_to_remove": [],
                "adoption_strategy": None,
                "failure_reason": "parse_failure",
            }
        parsed.setdefault("evaluation", {})
        parsed.setdefault("explanation", "")
        parsed.setdefault("new_query", None)
        parsed.setdefault("docs_to_remove", [])
        parsed.setdefault("adoption_strategy", None)
        return parsed

    def merge_and_filter_docs(
        self,
        old_docs: List[Dict],
        new_docs: List[Dict],
        docs_to_remove: List[int]
    ) -> List[Dict]:
        """
        Merge old and new documents, removing harmful/incorrect ones.

        Args:
            old_docs: Existing document list
            new_docs: Newly retrieved documents
            docs_to_remove: Indices of old_docs to remove

        Returns:
            Merged and filtered document list
        """
        # Filter out documents marked for removal
        filtered_old = [
            doc for idx, doc in enumerate(old_docs)
            if idx not in docs_to_remove
        ]

        # Merge with new documents, avoiding duplicates
        existing_ids = {doc.get("id") for doc in filtered_old}
        merged = filtered_old.copy()

        for doc in new_docs:
            doc_id = doc.get("id")
            if doc_id not in existing_ids:
                merged.append(doc)
                existing_ids.add(doc_id)

        # Limit to max_docs
        if len(merged) > self.max_docs:
            merged = merged[:self.max_docs]

        return merged

    def _generate_review_summary(self, review_history: List[Dict]) -> str:
        """Generate a summary of review history for forced adoption strategy."""
        if not review_history:
            return "No previous reviews."

        summary_parts = []
        for entry in review_history:
            eval_info = entry.get("evaluation", {})
            iteration = entry.get("iteration")
            iteration_label = iteration + 1 if isinstance(iteration, int) else iteration
            summary_parts.append(
                f"Iteration {iteration_label}: "
                f"Decision={entry['decision']}, "
                f"Coverage={entry.get('slot_coverage', {}).get('coverage', 'unknown')}, "
                f"Sufficiency={eval_info.get('sufficiency', 'unknown')}. "
                f"Explanation: {entry.get('explanation', 'N/A')}"
            )
        return "\n".join(summary_parts)

    def analyze_questions(self, questions: List[str], return_traces: bool = False):
        """Run QuestionAnalyzer once per question and return EvidenceState seeds."""
        prompts = [
            self.question_analyzer_prompt_template.get_string(question=question)
            for question in questions
        ]
        responses = self._batched_generate(self.reviewer, prompts)
        parsed_batch = self._parse_batch_with_retry(prompts, responses)

        states = []
        traces = []
        for question, parsed in zip(questions, parsed_batch):
            original_parsed = parsed
            if parsed.get("parse_failed"):
                parsed = {
                    "question_type": self._infer_question_type(question),
                    "answer_slots": self._fallback_slots(question),
                    "failure_reason": "question_analysis_parse_failure",
                    "explanation": parsed.get("parse_error", ""),
                }
            state = self._normalize_question_analysis(parsed, question)
            states.append(state)
            traces.append(self._question_analysis_trace(state, original_parsed))
        if return_traces:
            return states, traces
        return states

    def _build_evidence_review_prompt(
        self,
        question: str,
        docs: List[Dict],
        state: Dict[str, Any],
        round_id: int,
    ) -> str:
        return self.evidence_reviewer_prompt_template.get_string(
            question=question,
            formatted_docs=self.format_docs_for_review(docs),
            evidence_state=json.dumps(self._state_for_reviewer(state), ensure_ascii=False, indent=2),
            round_id=round_id,
        )

    def _state_for_reviewer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the reviewer state compact so JSON generation is less brittle."""
        compact = ReviewState.from_dict(state).compact_for_reviewer()
        if state.get("question_type") != "comparison":
            return compact
        if not self._comparison_branch_control_active(state):
            for slot in compact.get("answer_slots") or []:
                slot.pop("derived_output", None)
            return compact

        visible_slot_ids = {
            slot.get("slot_id")
            for slot in compact.get("answer_slots") or []
            if slot.get("slot_role") != "final_answer" and not slot.get("derived_output")
        }
        compact["answer_slots"] = [
            slot for slot in compact.get("answer_slots") or []
            if slot.get("slot_id") in visible_slot_ids
        ]
        compact["claims"] = [
            claim for claim in compact.get("claims") or []
            if claim.get("slot_id") in visible_slot_ids
        ]
        return compact

    def _retrieve_for_plans(
        self,
        retrieval_plans: List[Dict[str, Any]],
        iteration: Optional[int] = None,
        existing_docs: Optional[List[Dict]] = None,
        return_trace: bool = False,
    ):
        plans = [
            plan for plan in retrieval_plans[:self.max_slot_queries]
            if plan.get("query")
        ]
        queries = [plan.get("query") for plan in plans]
        if not queries:
            return ([], []) if return_trace else []

        if len(queries) == 1:
            results = [self.retriever.search(queries[0])]
        else:
            results = self.retriever.batch_search(queries)

        merged = []
        seen_ids = set()
        existing_ids = {
            self._safe_doc_id(doc, idx)
            for idx, doc in enumerate(existing_docs or [])
        }
        trace_seen_ids = set(existing_ids)
        traces = []
        for plan, docs in zip(plans, results):
            retrieved_ids = [self._safe_doc_id(doc, idx) for idx, doc in enumerate(docs)]
            new_ids = []
            duplicate_ids = []
            for doc_id in retrieved_ids:
                if doc_id in trace_seen_ids:
                    duplicate_ids.append(doc_id)
                else:
                    new_ids.append(doc_id)
                    trace_seen_ids.add(doc_id)
            traces.append({
                "iteration": iteration,
                "slot_id": plan.get("slot_id"),
                "query": plan.get("query"),
                "reason": plan.get("reason"),
                "retrieved_doc_ids": retrieved_ids,
                "retrieved_doc_titles": [self._doc_title(doc) for doc in docs],
                "retrieved_scores": [self._doc_score(doc) for doc in docs],
                "new_doc_ids_after_merge": new_ids,
                "duplicate_doc_ids": duplicate_ids,
                "new_doc_count": len(new_ids),
            })
            for doc in docs:
                doc_id = doc.get("id")
                if doc_id not in seen_ids:
                    merged.append(doc)
                    seen_ids.add(doc_id)
        if return_trace:
            return merged, traces
        return merged

    def _normalize_title(self, title: Any) -> str:
        return re.sub(r"\s+", " ", str(title or "").strip().lower())

    def _extract_gold_titles(self, metadata: Optional[Dict[str, Any]]) -> Tuple[str, List[str], int]:
        metadata = metadata or {}
        supporting = metadata.get("supporting_facts")
        if isinstance(supporting, dict) and supporting.get("title"):
            titles = [str(title) for title in supporting.get("title", []) if title]
            return "hotpotqa", titles, len(titles)

        decomposition = metadata.get("question_decomposition")
        if isinstance(decomposition, list):
            titles = []
            for step in decomposition:
                if not isinstance(step, dict):
                    continue
                support = step.get("support_paragraph") or {}
                title = support.get("title")
                if title:
                    titles.append(str(title))
            if titles:
                return "musique", titles, len(titles)

        return "unknown", [], 0

    def _hit_gold_titles(self, docs: List[Dict], gold_titles: List[str]) -> List[str]:
        normalized_gold = {self._normalize_title(title): title for title in gold_titles}
        hits = []
        for doc in docs or []:
            doc_title = self._normalize_title(self._doc_title(doc))
            if doc_title in normalized_gold and normalized_gold[doc_title] not in hits:
                hits.append(normalized_gold[doc_title])
        return hits

    def _build_gold_evidence_trace(
        self,
        metadata: Optional[Dict[str, Any]],
        initial_docs: List[Dict],
        final_docs: List[Dict],
        retrieval_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dataset_family, gold_titles, supporting_fact_count = self._extract_gold_titles(metadata)
        if dataset_family == "hotpotqa" and self.config["dataset_name"] == "2wikimultihopqa":
            dataset_family = "2wikimultihopqa"

        initial_hits = self._hit_gold_titles(initial_docs, gold_titles)
        final_hits = self._hit_gold_titles(final_docs, gold_titles)
        gold_set = set(gold_titles)
        initial_set = set(initial_hits)
        final_set = set(final_hits)

        retrieved_titles = []
        for trace in retrieval_trace or []:
            retrieved_titles.extend(trace.get("retrieved_doc_titles") or [])
        adaptive_hits = [
            title for title in gold_titles
            if title not in initial_set and title in {
                gold for gold in gold_titles
                if self._normalize_title(gold) in {self._normalize_title(t) for t in retrieved_titles}
            }
        ]

        denominator = len(gold_set)
        return {
            "dataset_family": dataset_family,
            "gold_titles": gold_titles,
            "initial_hit_titles": initial_hits,
            "final_hit_titles": final_hits,
            "initial_gold_title_recall": len(initial_set) / denominator if denominator else 0.0,
            "final_gold_title_recall": len(final_set) / denominator if denominator else 0.0,
            "adaptive_added_gold_titles": adaptive_hits,
            "lost_gold_titles_after_ranking": sorted(initial_set - final_set),
            "supporting_fact_count": supporting_fact_count,
            "initial_supporting_fact_hit_count": len(initial_set),
            "final_supporting_fact_hit_count": len(final_set),
        }

    def _record_review_metadata(
        self,
        state: Dict[str, Any],
        previous_state: Dict[str, Any],
        new_claim_count: int,
    ) -> None:
        previous_streak = int(previous_state.get("no_new_claim_streak") or 0)
        state["no_new_claim_streak"] = previous_streak + 1 if new_claim_count == 0 else 0
        state["query_history"] = list(previous_state.get("query_history") or state.get("query_history") or [])
        state["last_retrieval_plan"] = list(previous_state.get("last_retrieval_plan") or [])
        state["last_retrieval_new_doc_count"] = int(previous_state.get("last_retrieval_new_doc_count") or 0)
        state["recovery_triggered"] = bool(previous_state.get("recovery_triggered"))
        state["recovery_count"] = int(previous_state.get("recovery_count") or 0)
        state["recovery_reason"] = previous_state.get("recovery_reason")
        state["recovery_extra_round_pending"] = bool(previous_state.get("recovery_extra_round_pending"))
        state["recovery_extra_round_used"] = bool(previous_state.get("recovery_extra_round_used"))
        if new_claim_count == 0 and self._has_valid_retrieval_plan(state):
            state["recovery_triggered"] = True
            if state["no_new_claim_streak"] == 1:
                state["recovery_count"] += 1
            state["recovery_reason"] = state.get("recovery_reason") or "no_new_claim"

    def _record_retrieval_metadata(
        self,
        state: Dict[str, Any],
        retrieval_traces: List[Dict[str, Any]],
    ) -> None:
        if not retrieval_traces:
            return
        query_history = list(state.get("query_history") or [])
        last_retrieval_plan = []
        new_doc_count = 0
        for trace in retrieval_traces:
            query = str(trace.get("query") or "").strip()
            if query and query not in query_history:
                query_history.append(query)
            new_count = int(trace.get("new_doc_count") or 0)
            new_doc_count += new_count
            last_retrieval_plan.append({
                "slot_id": trace.get("slot_id"),
                "query": query,
                "reason": trace.get("reason"),
                "new_doc_count": new_count,
            })
        state["query_history"] = query_history
        state["last_retrieval_plan"] = last_retrieval_plan
        state["last_retrieval_new_doc_count"] = new_doc_count

    def _build_verifier_trace(self, state: Dict[str, Any], policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        trusted_claims = (policy or {}).get("trusted_claims") or []
        answer_supported = bool(state.get("direct_answer_supported"))
        return {
            "enabled": self.enable_answer_verifier,
            "answer_supported": answer_supported,
            "verified_claims": trusted_claims if self.enable_answer_verifier else [],
            "unsupported_claims": [],
            "verifier_failure_reason": None if self.enable_answer_verifier else "disabled",
        }

    def _build_final_trace(
        self,
        pred: str,
        golden_answers: List[str],
        state: Dict[str, Any],
        review_history: List[Dict[str, Any]],
        policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected_candidate = str((policy or {}).get("selected_answer_candidate") or "").strip()
        normalized_pred = self._normalize_answer_for_policy_match(pred)
        normalized_selected = self._normalize_answer_for_policy_match(selected_candidate)
        chain_trace = (
            (policy or {}).get("chain_grounding_trace")
            or state.get("chain_grounding_trace")
            or {}
        )
        relation_trace = (
            (policy or {}).get("relation_grounding_trace")
            or state.get("relation_grounding_trace")
            or {}
        )
        answer_decision = (
            (policy or {}).get("answer_decision")
            or state.get("answer_decision")
            or {}
        )
        used_forced_policy = state.get("decision") == "force_answer" or bool(state.get("failure_reason"))
        return {
            "pred": pred,
            "golden_answers": golden_answers,
            "answer_length": len(str(pred).split()),
            "used_forced_policy": used_forced_policy,
            "failure_reason": state.get("failure_reason"),
            "answer_type": state.get("answer_type"),
            "direct_answer_supported": bool(state.get("direct_answer_supported")),
            "answer_candidate": state.get("answer_candidate"),
            "selected_answer_candidate": selected_candidate or None,
            "answer_policy_trace": (policy or {}).get("answer_policy_trace", {}),
            "answer_intent": (policy or {}).get("answer_intent") or state.get("answer_intent", {}),
            "relation_grounding_trace": relation_trace,
            "answer_decision": answer_decision,
            "relation_grounded": bool(relation_trace.get("relation_grounded")),
            "semantic_coverage": relation_trace.get("semantic_coverage"),
            "subject_grounded": bool(relation_trace.get("subject_grounded")),
            "target_relation_match": bool(relation_trace.get("target_relation_match")),
            "candidate_role_valid": bool(relation_trace.get("candidate_role_valid")),
            "candidate_answer_type_match": bool(relation_trace.get("candidate_answer_type_match")),
            "candidate_is_question_entity": bool(relation_trace.get("candidate_is_question_entity")),
            "candidate_is_intermediate_entity": bool(relation_trace.get("candidate_is_intermediate_entity")),
            "grounded_but_wrong_relation": bool(relation_trace.get("grounded_but_wrong_relation")),
            "comparison_execution_status": relation_trace.get("comparison_execution_status"),
            "derived_answer_status": relation_trace.get("derived_answer_status"),
            "comparison_branches": relation_trace.get("comparison_branches") or [],
            "comparison_branch_status": relation_trace.get("comparison_branch_status"),
            "comparison_branch_executable": bool(
                relation_trace.get("comparison_branch_executable")
            ),
            "comparison_branch_failure_reason": relation_trace.get(
                "comparison_branch_failure_reason"
            ),
            "comparison_branch_build_time_ms": float(
                relation_trace.get("comparison_branch_build_time_ms") or 0.0
            ),
            "comparison_fallback_to_patch2": bool(
                state.get("comparison_fallback_to_patch2")
            ),
            "comparison_guidance_applied": bool(
                (policy or {}).get("comparison_guidance_applied")
                or state.get("comparison_guidance_applied")
            ),
            "comparison_output_contract": (policy or {}).get("comparison_output_contract") or {},
            "comparison_output_contract_available": bool(
                (policy or {}).get("comparison_output_contract_available")
            ),
            "comparison_output_contract_applied": bool(
                (policy or {}).get("comparison_output_contract_applied")
            ),
            "comparison_finalization_mode": (policy or {}).get("comparison_finalization_mode"),
            "comparison_commit_eligible": bool((policy or {}).get("comparison_commit_eligible")),
            "comparison_commit_blocked_reason": (policy or {}).get(
                "comparison_commit_blocked_reason"
            ),
            "comparison_commit_confidence": (policy or {}).get(
                "comparison_commit_confidence"
            ),
            "comparison_commit_relation_family": (policy or {}).get(
                "comparison_commit_relation_family"
            ),
            "comparison_commit_relation_value_type_valid": (policy or {}).get(
                "comparison_commit_relation_value_type_valid"
            ),
            "comparison_slot_associated_rejected": bool(
                (policy or {}).get("comparison_slot_associated_rejected")
            ),
            "comparison_final_candidate_matches_option": bool(
                (policy or {}).get("comparison_final_candidate_matches_option")
            ),
            "comparison_guided_generation": bool(
                (policy or {}).get("comparison_guided_generation")
            ),
            "comparison_final_commit_saved_generation": bool(
                (policy or {}).get("comparison_final_commit_saved_generation")
            ),
            "intent_source": relation_trace.get("intent_source"),
            "intent_conflict": bool(relation_trace.get("intent_conflict")),
            "operand_binding_status": relation_trace.get("operand_binding_status"),
            "comparison_lineage": relation_trace.get("comparison_lineage") or [],
            "strict_comparison_executable": bool(relation_trace.get("strict_comparison_executable")),
            "counterfactual_answer_candidate": relation_trace.get("counterfactual_answer_candidate"),
            "counterfactual_candidate_source": relation_trace.get("counterfactual_candidate_source"),
            "would_commit_under_v2_7": bool(relation_trace.get("would_commit_under_v2_7")),
            "control_action": relation_trace.get("control_action", "diagnostic_only"),
            "forced_by_relation_guard": bool(
                state.get("failure_reason") == "relation_ungrounded_final_slot"
            ),
            "generated_without_validated_candidate": bool(
                (policy or {}).get("generated_without_validated_candidate")
            ),
            "semantic_guidance_applied": bool((policy or {}).get("semantic_guidance_applied")),
            "diagnostic_candidate_leakage": bool((policy or {}).get("diagnostic_candidate_leakage")),
            "diagnostic_state_mutated": bool(relation_trace.get("diagnostic_state_mutated")),
            "diagnostic_query_mutated": bool(relation_trace.get("diagnostic_query_mutated")),
            "diagnostic_prompt_mutated": bool(relation_trace.get("diagnostic_prompt_mutated")),
            "diagnostic_answer_mutated": bool(relation_trace.get("diagnostic_answer_mutated")),
            "comparison_view_applied": bool(state.get("comparison_view_applied")),
            "derived_slots_hidden_from_reviewer": int(
                state.get("derived_slots_hidden_from_reviewer") or 0
            ),
            "derived_retrieval_plans_dropped": int(
                state.get("derived_retrieval_plans_dropped") or 0
            ),
            "non_comparison_prompt_mutated": False,
            "non_comparison_query_mutated": False,
            "non_comparison_output_mutated": False,
            "entity_binding_valid": bool(relation_trace.get("entity_binding_valid")),
            "binding_directions": relation_trace.get("binding_directions") or [],
            "original_question_type": state.get("original_question_type", state.get("question_type")),
            "effective_question_type": state.get("question_type"),
            "question_type_repaired": bool(state.get("question_type_repaired")),
            "question_type_repair_reason": state.get("question_type_repair_reason"),
            "commit_mode": answer_decision.get("commit_mode"),
            "commit_reason": answer_decision.get("commit_reason"),
            "committed_claim_ids": answer_decision.get("committed_claim_ids") or [],
            "chain_grounding_trace": chain_trace,
            "chain_coverage": chain_trace.get("chain_coverage"),
            "ungrounded_forced_answer": bool(
                used_forced_policy
                and chain_trace
                and not chain_trace.get("chain_grounded")
            ),
            "answer_policy_mismatch": bool(
                normalized_selected
                and normalized_pred
                and normalized_selected != normalized_pred
            ),
            "ready_guard_triggered": bool(state.get("ready_guard_triggered")),
            "guard_extra_round_used": bool(state.get("guard_extra_round_used")),
            "slot_coverage": self._slot_coverage(state),
            "retrieved_more": any(
                entry.get("retrieval_plan") or entry.get("new_doc_count", 0)
                for entry in review_history or []
            ),
        }

    def _normalize_answer_for_policy_match(self, answer: Any) -> str:
        text = str(answer or "").strip().lower()
        text = re.sub(r"\b(the|a|an)\b", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _finalize_policy_with_reviewer(
        self,
        question: str,
        docs: List[Dict],
        state: Dict[str, Any],
        review_history: List[Dict],
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Ask the reviewer for a final policy; fall back to deterministic policy on parse failure."""
        formatted_docs = self.format_docs_for_review(docs)
        review_summary = self._generate_review_summary(review_history)
        prompt = self.forced_adoption_prompt_template.get_string(
            question=question,
            formatted_docs=formatted_docs,
            evidence_state=json.dumps(state, ensure_ascii=False, indent=2),
            review_summary=review_summary,
        )
        response = self.reviewer.generate([prompt])[0]
        parsed, error = self._parse_json_response(response)
        if parsed is None:
            policy = self._build_adoption_policy(state, docs, failure_reason=failure_reason, question=question)
            policy["policy_parse_error"] = error
            return policy

        fallback = self._build_adoption_policy(state, docs, failure_reason=failure_reason, question=question)
        policy = {
            "trusted_claims": parsed.get("trusted_claims", fallback["trusted_claims"]),
            "answer_candidates": parsed.get("answer_candidates", fallback["answer_candidates"]),
            "selected_answer_candidate": parsed.get("selected_answer_candidate", fallback.get("selected_answer_candidate")),
            "candidate_source": parsed.get("candidate_source", fallback.get("candidate_source")),
            "answer_policy_trace": parsed.get("answer_policy_trace", fallback.get("answer_policy_trace")),
            "chain_grounding_trace": parsed.get("chain_grounding_trace", fallback.get("chain_grounding_trace")),
            "chain_coverage": parsed.get("chain_coverage", fallback.get("chain_coverage")),
            "answer_type": parsed.get("answer_type", fallback.get("answer_type")),
            "ignored_docs": parsed.get("ignored_docs", fallback["ignored_docs"]),
            "conflict_resolution": parsed.get("conflict_resolution", fallback["conflict_resolution"]),
            "must_use_slots": parsed.get("must_use_slots", fallback["must_use_slots"]),
            "insufficient_slots": parsed.get("insufficient_slots", fallback["insufficient_slots"]),
            "answer_style": parsed.get("answer_style", "short_answer"),
            "failure_reason": parsed.get("failure_reason", failure_reason),
        }
        for key in (
            "answer_intent",
            "relation_grounding_trace",
            "answer_decision",
            "comparison_output_contract",
            "comparison_output_contract_available",
            "comparison_output_contract_applied",
            "comparison_finalization_mode",
            "comparison_commit_eligible",
            "comparison_commit_blocked_reason",
            "comparison_commit_confidence",
            "comparison_commit_relation_family",
            "comparison_commit_relation_value_type_valid",
            "comparison_slot_associated_rejected",
            "comparison_final_candidate_matches_option",
            "comparison_guided_generation",
            "comparison_final_commit_saved_generation",
            "comparison_branches",
            "comparison_branch_status",
            "comparison_branch_executable",
            "comparison_branch_failure_reason",
            "comparison_guidance_applied",
            "comparison_fallback_to_patch2",
        ):
            policy[key] = fallback.get(key)
        if fallback.get("comparison_final_commit_saved_generation"):
            policy["selected_answer_candidate"] = fallback.get("selected_answer_candidate")
            policy["candidate_source"] = fallback.get("candidate_source")
        return policy

    def run_review_loop(
        self,
        question: str,
        initial_docs: List[Dict]
    ) -> Tuple[List[Dict], str, List[Dict], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Execute the review loop for a single question.

        Args:
            question: The input question
            initial_docs: Initially retrieved documents

        Returns:
            Tuple of (final_docs, rendered_policy, review_history, evidence_state, adoption_policy, diagnostic_trace)
        """
        current_docs = initial_docs
        review_history = []
        evidence_states, question_traces = self.analyze_questions([question], return_traces=True)
        evidence_state = evidence_states[0]
        diagnostic_trace = self._empty_diagnostic_trace()
        diagnostic_trace["question_analysis"] = question_traces[0]
        adoption_policy = None
        forced_reason = None

        for iteration in range(self.max_iterations + 1):
            if iteration >= self.max_iterations:
                if (
                    evidence_state.get("guard_extra_round_pending")
                    and not evidence_state.get("guard_extra_round_used")
                ):
                    evidence_state["guard_extra_round_pending"] = False
                    evidence_state["guard_extra_round_used"] = True
                else:
                    break
            elif evidence_state.get("guard_extra_round_pending"):
                evidence_state["guard_extra_round_pending"] = False

            num_docs_before_review = len(current_docs)
            reviewer_prompt = self._build_evidence_review_prompt(
                question, current_docs, evidence_state, iteration
            )
            reviewer_response = self.reviewer.generate([reviewer_prompt])[0]
            parsed_response = self._parse_batch_with_retry([reviewer_prompt], [reviewer_response])[0]
            decision_before_fallback = parsed_response.get("decision")
            previous_state = json.loads(json.dumps(evidence_state))
            evidence_state = self._merge_evidence_state(
                evidence_state, parsed_response, question, iteration
            )
            new_claim_count = self._count_new_claims(previous_state, evidence_state)
            self._record_review_metadata(evidence_state, previous_state, new_claim_count)
            coverage = self._slot_coverage(evidence_state)
            if (
                evidence_state.get("decision") == "continue_retrieval"
                and self._should_short_circuit_single_hop(evidence_state, coverage)
            ):
                evidence_state["decision"] = "ready_to_answer"
                evidence_state["retrieval_plan"] = []
            self._apply_chain_grounding_ready_guard(evidence_state, current_docs, question)
            self._mark_guard_extra_for_last_chance_retrieval(
                evidence_state,
                new_claim_count,
                iteration,
            )
            diagnostic_trace["review_trace"].append(self._review_step_trace(
                iteration,
                num_docs_before_review,
                parsed_response,
                evidence_state,
                coverage,
                new_claim_count,
                decision_before_fallback,
            ))

            review_history.append({
                "iteration": iteration,
                "num_docs": len(current_docs),
                "decision": evidence_state["decision"],
                "slot_coverage": coverage,
                "new_claim_count": new_claim_count,
                "retrieval_plan": evidence_state.get("retrieval_plan", []),
                "failure_reason": evidence_state.get("failure_reason"),
                "explanation": evidence_state.get("explanation", ""),
            })

            docs_to_remove = evidence_state.get("docs_to_remove", [])
            if docs_to_remove:
                current_docs = [
                    doc for idx, doc in enumerate(current_docs)
                    if idx not in docs_to_remove
                ]

            if evidence_state["decision"] == "ready_to_answer":
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._build_adoption_policy(evidence_state, current_docs, question=question)
                break

            if evidence_state["decision"] in {"force_answer", "review_failed"}:
                forced_reason = evidence_state.get("failure_reason") or evidence_state["decision"]
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._build_adoption_policy(
                    evidence_state,
                    current_docs,
                    failure_reason=forced_reason,
                    question=question,
                )
                break

            if self._should_force_no_new_claim(evidence_state, new_claim_count, iteration):
                forced_reason = "no_new_claim"
                evidence_state["decision"] = "force_answer"
                evidence_state["failure_reason"] = forced_reason
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._build_adoption_policy(
                    evidence_state,
                    current_docs,
                    failure_reason=forced_reason,
                    question=question,
                )
                review_history[-1]["decision"] = "force_answer"
                review_history[-1]["failure_reason"] = forced_reason
                diagnostic_trace["review_trace"][-1]["decision_after_merge"] = "force_answer"
                diagnostic_trace["review_trace"][-1]["failure_reason"] = forced_reason
                break

            retrieval_plan = evidence_state.get("retrieval_plan", [])
            if self._has_valid_retrieval_plan(evidence_state):
                new_docs, retrieval_trace = self._retrieve_for_plans(
                    retrieval_plan,
                    iteration=iteration,
                    existing_docs=current_docs,
                    return_trace=True,
                )
                diagnostic_trace["retrieval_trace"].extend(retrieval_trace)
                self._record_retrieval_metadata(evidence_state, retrieval_trace)
                current_docs = self.merge_and_filter_docs(current_docs, new_docs, [])
                review_history[-1]["new_doc_count"] = sum(
                    trace.get("new_doc_count", 0) for trace in retrieval_trace
                )
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
            else:
                forced_reason = "empty_retrieval_plan"
                evidence_state["decision"] = "force_answer"
                evidence_state["failure_reason"] = forced_reason
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._build_adoption_policy(
                    evidence_state,
                    current_docs,
                    failure_reason=forced_reason,
                    question=question,
                )
                diagnostic_trace["review_trace"][-1]["decision_after_merge"] = "force_answer"
                diagnostic_trace["review_trace"][-1]["failure_reason"] = forced_reason
                break

        if adoption_policy is None:
            forced_reason = forced_reason or "max_rounds"
            evidence_state["decision"] = "force_answer"
            evidence_state["failure_reason"] = forced_reason
            current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
            rank_trace["iteration"] = "forced"
            diagnostic_trace["ranking_trace"].append(rank_trace)
            adoption_policy = self._build_adoption_policy(
                evidence_state,
                current_docs,
                failure_reason=forced_reason,
                question=question,
            )
            review_history.append({
                "iteration": "forced",
                "num_docs": len(current_docs),
                "decision": "force_answer",
                "slot_coverage": self._slot_coverage(evidence_state),
                "new_claim_count": 0,
                "retrieval_plan": [],
                "failure_reason": forced_reason,
                "explanation": "Max review iterations reached.",
            })

        rendered_policy = self._render_adoption_policy(adoption_policy)
        self._attach_policy_diagnostics(diagnostic_trace, adoption_policy)
        return current_docs, rendered_policy, review_history, evidence_state, adoption_policy, diagnostic_trace

    def run_item(self, item) -> None:
        """Process a single item through the reviewer pipeline."""
        question = item.question

        # Step 1: Initial retrieval
        initial_docs = self.retriever.search(question)
        item.update_output("initial_retrieval_result", initial_docs)

        # Step 2: Review loop
        final_docs, adoption_strategy, review_history, evidence_state, adoption_policy, diagnostic_trace = self.run_review_loop(
            question, initial_docs
        )
        diagnostic_trace["gold_evidence_trace"] = self._build_gold_evidence_trace(
            item.metadata,
            initial_docs,
            final_docs,
            diagnostic_trace.get("retrieval_trace", []),
        )
        verifier_trace = self._build_verifier_trace(evidence_state, adoption_policy)
        self._attach_policy_diagnostics(diagnostic_trace, adoption_policy)
        diagnostic_trace["verifier_trace"] = verifier_trace

        # Store intermediate results
        item.update_output("retrieval_result", final_docs)
        item.update_output("adoption_strategy", adoption_strategy)
        item.update_output("adoption_policy", adoption_policy)
        item.update_output("verifier_trace", verifier_trace)
        item.update_output("answer_supported", verifier_trace.get("answer_supported"))
        item.update_output("evidence_state", evidence_state)
        item.update_output("slot_coverage", self._slot_coverage(evidence_state))
        item.update_output("failure_reason", evidence_state.get("failure_reason"))
        item.update_output("review_history", review_history)

        # Step 3: Final generation with adoption strategy
        formatted_reference = self.format_docs_for_generation(final_docs)

        # Create final prompt with adoption strategy
        final_prompt = self.prompt_template.get_string(
            question=question,
            reference=formatted_reference,
            adoption_strategy=adoption_strategy
        )

        item.update_output("prompt", final_prompt)

        # Deterministic decisions bypass the final generator.
        pred = self._generate_final_answers(
            [final_prompt],
            [adoption_policy],
            [evidence_state],
        )[0]
        item.update_output("pred", pred)
        diagnostic_trace["final_trace"] = self._build_final_trace(
            pred,
            item.golden_answers,
            evidence_state,
            review_history,
            adoption_policy,
        )
        item.update_output("diagnostic_trace", diagnostic_trace)

    def run_batch(self, dataset) -> None:
        """
        Process a batch of items with EvidenceState-controlled review.

        Flow:
        - Initial retrieval
        - One question analysis pass to create answer slots
        - Layered evidence review and slot-based re-retrieval
        - Structured adoption policy rendering
        - Final batch generation
        """
        questions = dataset.question
        n_items = len(questions)

        print(f"\n{'='*60}")
        print(f"Reviewer Pipeline: {n_items} items, max {self.max_iterations} iterations")
        print(f"{'='*60}")

        # Step 1: Batch initial retrieval
        print(f"\n[1/5] Initial retrieval...")
        initial_retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("initial_retrieval_result", initial_retrieval_results)
        print(f"  Done: retrieved docs for {n_items} questions")

        print(f"\n[2/5] Question analysis...")
        evidence_states, question_traces = self.analyze_questions(questions, return_traces=True)
        diagnostic_traces = [self._empty_diagnostic_trace() for _ in range(n_items)]
        for idx, trace in enumerate(question_traces):
            diagnostic_traces[idx]["question_analysis"] = trace
        print(f"  Done: initialized EvidenceState for {n_items} questions")

        current_docs = [docs.copy() if docs else [] for docs in initial_retrieval_results]
        adoption_policies = [None] * n_items
        adoption_strategies = [None] * n_items
        review_histories = [[] for _ in range(n_items)]
        active_mask = [True] * n_items  # True = still needs processing

        print(f"\n[3/5] Evidence review loop...")

        for iteration in range(self.max_iterations + 1):
            if iteration >= self.max_iterations:
                active_indices = [
                    i for i, active in enumerate(active_mask)
                    if active
                    and evidence_states[i].get("guard_extra_round_pending")
                    and not evidence_states[i].get("guard_extra_round_used")
                ]
                for idx in active_indices:
                    evidence_states[idx]["guard_extra_round_pending"] = False
                    evidence_states[idx]["guard_extra_round_used"] = True
            else:
                active_indices = [i for i, active in enumerate(active_mask) if active]
                for idx in active_indices:
                    if evidence_states[idx].get("guard_extra_round_pending"):
                        evidence_states[idx]["guard_extra_round_pending"] = False
            if not active_indices:
                print(f"  All items completed at iteration {iteration}")
                break

            n_active = len(active_indices)
            n_completed = n_items - n_active
            iter_label = f"{iteration + 1}/{self.max_iterations}"
            if iteration >= self.max_iterations:
                iter_label = "guard-extra"
            print(f"\n  Iter {iter_label}: {n_active} active, {n_completed} done")

            reviewer_prompts = []
            for idx in active_indices:
                reviewer_prompts.append(
                    self._build_evidence_review_prompt(
                        questions[idx],
                        current_docs[idx],
                        evidence_states[idx],
                        iteration,
                    )
                )

            print(f"    Reviewing {n_active} items...")
            reviewer_responses = self._batched_generate(self.reviewer, reviewer_prompts)
            parsed_responses = self._parse_batch_with_retry(reviewer_prompts, reviewer_responses)

            retrieval_requests = []
            n_ready = 0
            n_need_more = 0
            n_forced = 0
            n_failed = 0

            for batch_idx, idx in enumerate(active_indices):
                parsed = parsed_responses[batch_idx]
                num_docs_before_review = len(current_docs[idx])
                decision_before_fallback = parsed.get("decision")
                previous_state = json.loads(json.dumps(evidence_states[idx]))
                evidence_states[idx] = self._merge_evidence_state(
                    evidence_states[idx],
                    parsed,
                    questions[idx],
                    iteration,
                )
                new_claim_count = self._count_new_claims(previous_state, evidence_states[idx])
                self._record_review_metadata(evidence_states[idx], previous_state, new_claim_count)
                coverage = self._slot_coverage(evidence_states[idx])
                if (
                    evidence_states[idx].get("decision") == "continue_retrieval"
                    and self._should_short_circuit_single_hop(evidence_states[idx], coverage)
                ):
                    evidence_states[idx]["decision"] = "ready_to_answer"
                    evidence_states[idx]["retrieval_plan"] = []
                self._apply_chain_grounding_ready_guard(
                    evidence_states[idx],
                    current_docs[idx],
                    questions[idx],
                )
                self._mark_guard_extra_for_last_chance_retrieval(
                    evidence_states[idx],
                    new_claim_count,
                    iteration,
                )
                diagnostic_traces[idx]["review_trace"].append(self._review_step_trace(
                    iteration,
                    num_docs_before_review,
                    parsed,
                    evidence_states[idx],
                    coverage,
                    new_claim_count,
                    decision_before_fallback,
                ))

                review_histories[idx].append({
                    "iteration": iteration,
                    "num_docs": len(current_docs[idx]),
                    "decision": evidence_states[idx]["decision"],
                    "slot_coverage": coverage,
                    "new_claim_count": new_claim_count,
                    "retrieval_plan": evidence_states[idx].get("retrieval_plan", []),
                    "failure_reason": evidence_states[idx].get("failure_reason"),
                    "explanation": evidence_states[idx].get("explanation", ""),
                })

                docs_to_remove = evidence_states[idx].get("docs_to_remove", [])
                if docs_to_remove:
                    current_docs[idx] = [
                        doc for i, doc in enumerate(current_docs[idx])
                        if i not in docs_to_remove
                    ]

                decision = evidence_states[idx]["decision"]
                if decision == "ready_to_answer":
                    current_docs[idx], rank_trace = self._rank_docs_by_state(
                        current_docs[idx], evidence_states[idx], return_trace=True
                    )
                    rank_trace["iteration"] = iteration
                    diagnostic_traces[idx]["ranking_trace"].append(rank_trace)
                    adoption_policies[idx] = self._build_adoption_policy(
                        evidence_states[idx],
                        current_docs[idx],
                        question=questions[idx],
                    )
                    adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                    active_mask[idx] = False
                    n_ready += 1
                elif decision in {"force_answer", "review_failed"}:
                    reason = evidence_states[idx].get("failure_reason") or decision
                    current_docs[idx], rank_trace = self._rank_docs_by_state(
                        current_docs[idx], evidence_states[idx], return_trace=True
                    )
                    rank_trace["iteration"] = iteration
                    diagnostic_traces[idx]["ranking_trace"].append(rank_trace)
                    adoption_policies[idx] = self._build_adoption_policy(
                        evidence_states[idx],
                        current_docs[idx],
                        failure_reason=reason,
                        question=questions[idx],
                    )
                    adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                    active_mask[idx] = False
                    n_forced += 1
                    if decision == "review_failed":
                        n_failed += 1
                elif self._should_force_no_new_claim(evidence_states[idx], new_claim_count, iteration):
                    reason = "no_new_claim"
                    evidence_states[idx]["decision"] = "force_answer"
                    evidence_states[idx]["failure_reason"] = reason
                    current_docs[idx], rank_trace = self._rank_docs_by_state(
                        current_docs[idx], evidence_states[idx], return_trace=True
                    )
                    rank_trace["iteration"] = iteration
                    diagnostic_traces[idx]["ranking_trace"].append(rank_trace)
                    adoption_policies[idx] = self._build_adoption_policy(
                        evidence_states[idx],
                        current_docs[idx],
                        failure_reason=reason,
                        question=questions[idx],
                    )
                    adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                    review_histories[idx][-1]["decision"] = "force_answer"
                    review_histories[idx][-1]["failure_reason"] = reason
                    diagnostic_traces[idx]["review_trace"][-1]["decision_after_merge"] = "force_answer"
                    diagnostic_traces[idx]["review_trace"][-1]["failure_reason"] = reason
                    active_mask[idx] = False
                    n_forced += 1
                else:
                    retrieval_plan = evidence_states[idx].get("retrieval_plan", [])
                    if self._has_valid_retrieval_plan(evidence_states[idx]):
                        retrieval_requests.append((idx, retrieval_plan))
                    else:
                        reason = "empty_retrieval_plan"
                        evidence_states[idx]["decision"] = "force_answer"
                        evidence_states[idx]["failure_reason"] = reason
                        current_docs[idx], rank_trace = self._rank_docs_by_state(
                            current_docs[idx], evidence_states[idx], return_trace=True
                        )
                        rank_trace["iteration"] = iteration
                        diagnostic_traces[idx]["ranking_trace"].append(rank_trace)
                        adoption_policies[idx] = self._build_adoption_policy(
                            evidence_states[idx],
                            current_docs[idx],
                            failure_reason=reason,
                            question=questions[idx],
                        )
                        adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                        review_histories[idx][-1]["decision"] = "force_answer"
                        review_histories[idx][-1]["failure_reason"] = reason
                        diagnostic_traces[idx]["review_trace"][-1]["decision_after_merge"] = "force_answer"
                        diagnostic_traces[idx]["review_trace"][-1]["failure_reason"] = reason
                        active_mask[idx] = False
                        n_forced += 1
                    n_need_more += 1

            print(
                f"    Results: {n_ready} ready, {n_need_more} need more, "
                f"{n_forced} forced, {n_failed} review failed"
            )

            if retrieval_requests:
                flat_queries = []
                query_owner = []
                query_plans = []
                for idx, retrieval_plan in retrieval_requests:
                    for plan in retrieval_plan[:self.max_slot_queries]:
                        query = plan.get("query")
                        if query:
                            flat_queries.append(query)
                            query_owner.append(idx)
                            query_plans.append(plan)

                if flat_queries:
                    print(f"    Slot re-retrieving with {len(flat_queries)} queries...")
                    flat_results = self.retriever.batch_search(flat_queries)
                    docs_by_idx = {}
                    retrieval_traces_by_idx = {}
                    trace_seen_ids = {
                        idx: {self._safe_doc_id(doc, doc_idx) for doc_idx, doc in enumerate(current_docs[idx])}
                        for idx, _ in retrieval_requests
                    }
                    for owner_idx, plan, docs in zip(query_owner, query_plans, flat_results):
                        docs_by_idx.setdefault(owner_idx, []).extend(docs)
                        retrieved_ids = [self._safe_doc_id(doc, doc_idx) for doc_idx, doc in enumerate(docs)]
                        new_ids = []
                        duplicate_ids = []
                        for doc_id in retrieved_ids:
                            if doc_id in trace_seen_ids[owner_idx]:
                                duplicate_ids.append(doc_id)
                            else:
                                new_ids.append(doc_id)
                                trace_seen_ids[owner_idx].add(doc_id)
                        retrieval_trace = {
                            "iteration": iteration,
                            "slot_id": plan.get("slot_id"),
                            "query": plan.get("query"),
                            "reason": plan.get("reason"),
                            "retrieved_doc_ids": retrieved_ids,
                            "retrieved_doc_titles": [self._doc_title(doc) for doc in docs],
                            "retrieved_scores": [self._doc_score(doc) for doc in docs],
                            "new_doc_ids_after_merge": new_ids,
                            "duplicate_doc_ids": duplicate_ids,
                            "new_doc_count": len(new_ids),
                        }
                        diagnostic_traces[owner_idx]["retrieval_trace"].append(retrieval_trace)
                        retrieval_traces_by_idx.setdefault(owner_idx, []).append(retrieval_trace)

                    for idx, new_docs in docs_by_idx.items():
                        before_ids = {doc.get("id") for doc in current_docs[idx]}
                        current_docs[idx] = self.merge_and_filter_docs(current_docs[idx], new_docs, [])
                        after_ids = {doc.get("id") for doc in current_docs[idx]}
                        review_histories[idx][-1]["new_doc_count"] = len(after_ids - before_ids)
                        self._record_retrieval_metadata(
                            evidence_states[idx],
                            retrieval_traces_by_idx.get(idx, []),
                        )
                        current_docs[idx], rank_trace = self._rank_docs_by_state(
                            current_docs[idx], evidence_states[idx], return_trace=True
                        )
                        rank_trace["iteration"] = iteration
                        diagnostic_traces[idx]["ranking_trace"].append(rank_trace)

        remaining_indices = [i for i, active in enumerate(active_mask) if active]
        if remaining_indices:
            print(f"\n[4/5] Forcing policy for {len(remaining_indices)} remaining items...")
            for idx in remaining_indices:
                reason = "max_rounds"
                evidence_states[idx]["decision"] = "force_answer"
                evidence_states[idx]["failure_reason"] = reason
                current_docs[idx], rank_trace = self._rank_docs_by_state(
                    current_docs[idx], evidence_states[idx], return_trace=True
                )
                rank_trace["iteration"] = "forced"
                diagnostic_traces[idx]["ranking_trace"].append(rank_trace)
                adoption_policies[idx] = self._build_adoption_policy(
                    evidence_states[idx],
                    current_docs[idx],
                    failure_reason=reason,
                    question=questions[idx],
                )
                adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                active_mask[idx] = False
                review_histories[idx].append({
                    "iteration": "forced",
                    "num_docs": len(current_docs[idx]),
                    "decision": "force_answer",
                    "slot_coverage": self._slot_coverage(evidence_states[idx]),
                    "new_claim_count": 0,
                    "retrieval_plan": [],
                    "failure_reason": reason,
                    "explanation": "Max review iterations reached.",
                })
            print(f"  Done: {len(remaining_indices)} forced policies")
        else:
            print(f"\n[4/5] All items completed within limit")

        for idx in range(n_items):
            if adoption_policies[idx] is None:
                reason = evidence_states[idx].get("failure_reason") or "missing_policy"
                adoption_policies[idx] = self._build_adoption_policy(
                    evidence_states[idx],
                    current_docs[idx],
                    failure_reason=reason,
                    question=questions[idx],
                )
            if adoption_strategies[idx] is None:
                adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
            self._attach_policy_diagnostics(diagnostic_traces[idx], adoption_policies[idx])

        slot_coverages = [self._slot_coverage(state) for state in evidence_states]
        failure_reasons = [state.get("failure_reason") for state in evidence_states]

        dataset.update_output("retrieval_result", current_docs)
        dataset.update_output("evidence_state", evidence_states)
        dataset.update_output("adoption_policy", adoption_policies)
        dataset.update_output("adoption_strategy", adoption_strategies)
        dataset.update_output("slot_coverage", slot_coverages)
        dataset.update_output("failure_reason", failure_reasons)
        dataset.update_output("review_history", review_histories)
        for idx in range(n_items):
            diagnostic_traces[idx]["gold_evidence_trace"] = self._build_gold_evidence_trace(
                dataset.data[idx].metadata,
                initial_retrieval_results[idx],
                current_docs[idx],
                diagnostic_traces[idx].get("retrieval_trace", []),
            )
        verifier_traces = []
        answer_supported = []
        for idx in range(n_items):
            verifier_trace = self._build_verifier_trace(evidence_states[idx], adoption_policies[idx])
            diagnostic_traces[idx]["verifier_trace"] = verifier_trace
            verifier_traces.append(verifier_trace)
            answer_supported.append(verifier_trace.get("answer_supported"))
        dataset.update_output("verifier_trace", verifier_traces)
        dataset.update_output("answer_supported", answer_supported)

        print(f"\n[5/5] Generating {n_items} final answers...")

        all_prompts = []
        for idx in range(n_items):
            formatted_reference = self.format_docs_for_generation(current_docs[idx])
            prompt = self.prompt_template.get_string(
                question=questions[idx],
                reference=formatted_reference,
                adoption_strategy=adoption_strategies[idx]
            )
            all_prompts.append(prompt)

        dataset.update_output("prompt", all_prompts)

        pred_answers = self._generate_final_answers(
            all_prompts,
            adoption_policies,
            evidence_states,
        )
        dataset.update_output("pred", pred_answers)
        for idx, pred in enumerate(pred_answers):
            diagnostic_traces[idx]["final_trace"] = self._build_final_trace(
                pred,
                dataset.data[idx].golden_answers,
                evidence_states[idx],
                review_histories[idx],
                adoption_policies[idx],
            )
        dataset.update_output("diagnostic_trace", diagnostic_traces)

        print(f"\n{'='*60}")
        print(f"Completed: {n_items} items")
        print(f"{'='*60}\n")

    def run(
        self,
        dataset,
        do_eval: bool = True,
        pred_process_fun=None,
        sample_size: Optional[int] = None,
    ):
        """
        Run the reviewer pipeline on the dataset.

        Args:
            dataset: Input dataset
            do_eval: Whether to run evaluation
            pred_process_fun: Optional post-processing function for predictions
            sample_size: Optional limit on number of samples to process.
                         If None, process all samples.

        Returns:
            Processed dataset with predictions and intermediate results
        """
        # Limit dataset size if sample_size is specified
        if sample_size is not None and sample_size > 0:
            original_size = len(dataset)
            # dataset[:n] returns list, need to wrap back into Dataset
            from flashrag.dataset import Dataset as FlashRAGDataset
            sliced_data = dataset.data[:sample_size]
            dataset = FlashRAGDataset(config=dataset.config, data=sliced_data)
            print(f"Using {len(dataset)}/{original_size} samples (sample_size={sample_size})")

        self._write_manifest(dataset_size=len(dataset))
        self.run_batch(dataset)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ReviewerPipelineWithSeparateReviewer(ReviewerPipeline):
    """
    Reviewer Pipeline with a separate LLM model for review.

    This variant allows using different models for review and generation,
    e.g., using Qwen3 for review and LLaMA3 for generation.

    Args:
        config: FlashRAG configuration
        reviewer_config: Configuration for the reviewer LLM
        ... (other args same as ReviewerPipeline)
    """

    def __init__(
        self,
        config,
        reviewer_config: Optional[Dict] = None,
        prompt_template: Optional[PromptTemplate] = None,
        max_iterations: int = 3,
        retriever=None,
        generator=None,
        max_docs: int = 10,
        batch_size: int = 256,
    ):
        # First initialize the reviewer if separate config provided
        reviewer = None
        if reviewer_config is not None:
            from flashrag.config import Config
            # Create a temporary config for reviewer
            reviewer_cfg = Config(config_dict=reviewer_config)
            reviewer = get_generator(reviewer_cfg)

        super().__init__(
            config=config,
            prompt_template=prompt_template,
            max_iterations=max_iterations,
            retriever=retriever,
            generator=generator,
            reviewer=reviewer,
            max_docs=max_docs,
            batch_size=batch_size,
        )
