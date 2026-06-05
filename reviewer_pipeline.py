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
import re
from typing import List, Dict, Any, Tuple, Optional

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate


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
    "answer_slots": [
        {{
            "slot_id": "slot_1",
            "description": "specific fact needed to answer",
            "slot_role": "intermediate_entity | final_answer | comparison_operand | direct_answer",
            "depends_on": [],
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
    "answer_slots": [
        {{
            "slot_id": "slot_1",
            "description": "specific fact needed to answer",
            "slot_role": "intermediate_entity | final_answer | comparison_operand | direct_answer",
            "depends_on": [],
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

    def _parse_json_response(self, response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse a JSON object without converting failures into READY decisions."""
        try:
            return json.loads(self._extract_json_text(response)), None
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None, str(e)

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
            "question_analysis": {
                "question_type": "",
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
            "gold_evidence_trace": {},
            "final_trace": {},
        }

    def _question_analysis_trace(self, state: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        slots = state.get("answer_slots", [])
        raw_response = None
        if parsed.get("parse_failed") or parsed.get("retry_failed") or parsed.get("repaired"):
            raw_response = parsed.get("raw_response") or parsed.get("retry_response")
        return {
            "question_type": state.get("question_type", ""),
            "slot_count": len(slots),
            "slot_descriptions": [slot.get("description", "") for slot in slots],
            "slot_roles": [slot.get("slot_role", "") for slot in slots],
            "parse_failed": bool(parsed.get("parse_failed") or parsed.get("retry_failed")),
            "parse_error": parsed.get("parse_error") or parsed.get("retry_error"),
            "raw_response_on_failure": self._truncate_text(raw_response),
            "repaired": bool(parsed.get("repaired")),
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
            "answer_candidate": state.get("answer_candidate"),
            "ready_guard_triggered": bool(state.get("ready_guard_triggered")),
            "failure_reason": state.get("failure_reason"),
            "parse_failed": bool(parsed.get("parse_failed") or parsed.get("retry_failed")),
            "parse_error": parsed.get("parse_error") or parsed.get("retry_error"),
            "repaired": bool(parsed.get("repaired")),
            "raw_response_on_failure": self._truncate_text(raw_response),
        }

    def _infer_question_type(self, question: str) -> str:
        q = question.lower()
        comparison_markers = ["both", "same", "older", "younger", "larger", "smaller", "more", "less", "compare"]
        bridge_markers = ["who", "what", "where", "when", "which"]
        if any(marker in q for marker in comparison_markers):
            return "comparison"
        if any(marker in q for marker in bridge_markers) and (" of " in q or " by " in q or " in " in q):
            return "bridge"
        return "single_hop"

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
            return self._fallback_slots(question, question_type)

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
            })

        return normalized or self._fallback_slots(question, question_type)

    def _default_slot_role(self, question_type: Optional[str], idx: int, slot_count: int) -> str:
        if question_type == "bridge":
            return "final_answer" if idx == slot_count - 1 else "intermediate_entity"
        if question_type == "comparison":
            return "final_answer" if idx == slot_count - 1 else "comparison_operand"
        return "direct_answer"

    def _normalize_question_analysis(self, parsed: Dict[str, Any], question: str) -> Dict[str, Any]:
        valid_types = {"single_hop", "bridge", "comparison", "ambiguous"}
        question_type = parsed.get("question_type")
        if question_type not in valid_types:
            question_type = self._infer_question_type(question)

        return {
            "question_type": question_type,
            "answer_slots": self._normalize_slots(parsed.get("answer_slots"), question, question_type),
            "claims": [],
            "retrieval_plan": [],
            "decision": "continue_retrieval",
            "direct_answer_supported": False,
            "answer_candidate": None,
            "answer_candidate_claim_ids": [],
            "failure_reason": parsed.get("failure_reason"),
            "explanation": parsed.get("explanation", ""),
            "round_id": 0,
        }

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
            normalized.append({
                "claim_id": str(claim.get("claim_id") or f"c{idx + 1}"),
                "doc_id": str(claim.get("doc_id") or ""),
                "slot_id": slot_id,
                "text": text,
                "support_status": support_status,
                "round_id": claim.get("round_id", round_id),
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
            state["decision"] = "ready_to_answer"
            state["failure_reason"] = "parse_failure_fallback"
            state["retrieval_plan"] = []
            state["round_id"] = round_id
            return state

        slots = self._normalize_slots(
            parsed.get("answer_slots", previous_state.get("answer_slots")),
            question,
            previous_state.get("question_type"),
        )
        valid_slot_ids = {slot["slot_id"] for slot in slots}

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

        retrieval_plan = self._normalize_retrieval_plan(parsed.get("retrieval_plan"), valid_slot_ids)
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
        answer_candidate = str(parsed.get("answer_candidate") or "").strip() or None
        answer_candidate_claim_ids = list(parsed.get("answer_candidate_claim_ids") or [])
        final_answer_slot_ids = self._final_answer_slot_ids(slots)
        candidate_claim_ids = {
            claim.get("claim_id")
            for claim in merged_claims
            if claim.get("slot_id") in final_answer_slot_ids
            and claim.get("support_status") == "support"
        }
        if answer_candidate and answer_candidate_claim_ids:
            direct_answer_supported = bool(candidate_claim_ids.intersection(answer_candidate_claim_ids))

        ready_guard_triggered = False
        if decision == "ready_to_answer" and not direct_answer_supported:
            ready_guard_triggered = True
            if not retrieval_plan and previous_state.get("question_type") == "bridge":
                retrieval_plan = self._build_bridge_fallback_plan(question, slots, merged_claims)
            decision = "continue_retrieval" if retrieval_plan else "force_answer"

        if decision == "continue_retrieval" and not retrieval_plan:
            unresolved = [slot for slot in slots if slot["status"] in {"missing", "partial", "conflicted"}]
            if not unresolved:
                decision = "ready_to_answer" if direct_answer_supported else "force_answer"
            else:
                decision = "force_answer"

        return {
            "question_type": previous_state.get("question_type", self._infer_question_type(question)),
            "answer_slots": slots,
            "claims": merged_claims,
            "retrieval_plan": retrieval_plan,
            "decision": decision,
            "direct_answer_supported": direct_answer_supported,
            "answer_candidate": answer_candidate,
            "answer_candidate_claim_ids": answer_candidate_claim_ids,
            "ready_guard_triggered": ready_guard_triggered,
            "docs_to_remove": self._sanitize_doc_indices(parsed.get("docs_to_remove", [])),
            "ignored_docs": self._sanitize_doc_indices(parsed.get("ignored_docs", [])),
            "failure_reason": parsed.get("failure_reason"),
            "explanation": parsed.get("explanation", ""),
            "round_id": round_id,
        }

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

    def _slot_coverage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        slots = state.get("answer_slots", [])
        total = len(slots)
        supported = sum(1 for slot in slots if slot.get("status") == "supported")
        conflicted = sum(1 for slot in slots if slot.get("status") == "conflicted")
        missing = sum(1 for slot in slots if slot.get("status") in {"missing", "partial"})
        return {
            "total": total,
            "supported": supported,
            "conflicted": conflicted,
            "missing": missing,
            "coverage": supported / total if total else 0.0,
        }

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
    ) -> Dict[str, Any]:
        trusted_claims = [
            claim for claim in state.get("claims", [])
            if claim.get("support_status") == "support"
        ]
        conflicted_claims = [
            claim for claim in state.get("claims", [])
            if claim.get("support_status") == "contradict"
        ]
        insufficient_slots = [
            slot["slot_id"] for slot in state.get("answer_slots", [])
            if slot.get("status") in {"missing", "partial", "conflicted"}
        ]
        must_use_slots = [
            slot["slot_id"] for slot in state.get("answer_slots", [])
            if slot.get("status") == "supported"
        ]
        answer_candidates = []
        answer_candidate = str(state.get("answer_candidate") or "").strip()
        if answer_candidate:
            answer_candidates.append(answer_candidate)
        final_answer_slot_ids = self._final_answer_slot_ids(state.get("answer_slots", []))
        for claim in trusted_claims:
            if claim.get("slot_id") not in final_answer_slot_ids:
                continue
            candidate = self._claim_to_short_answer_candidate(claim.get("text", ""))
            if candidate and candidate not in answer_candidates:
                answer_candidates.append(candidate)
        return {
            "trusted_claims": trusted_claims,
            "answer_candidates": answer_candidates[:8],
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
        answer_candidates = policy.get("answer_candidates") or []
        insufficient_slots = policy.get("insufficient_slots") or []
        ignored_docs = policy.get("ignored_docs") or []
        conflict_resolution = policy.get("conflict_resolution") or []

        lines = ["Supported facts to use:"]
        if trusted_claims:
            for claim in trusted_claims[:12]:
                text = str(claim.get("text") or "").strip()
                doc_id = claim.get("doc_id")
                if text:
                    lines.append(f"- {text}" + (f" (source document id: {doc_id})" if doc_id else ""))
        else:
            lines.append("- No explicit supported claim was extracted; rely on the most relevant document text.")

        if answer_candidates:
            lines.append("Preferred short answer candidates:")
            for candidate in answer_candidates[:8]:
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
        answer_candidates = [
            str(candidate).strip()
            for candidate in policy.get("answer_candidates") or []
            if str(candidate).strip()
        ]
        if answer_candidates:
            return answer_candidates[0]
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

    def _claim_to_short_answer_candidate(self, claim_text: str) -> Optional[str]:
        text = str(claim_text or "").strip()
        if not text:
            return None

        quoted = re.findall(r'"([^"\n]{2,80})"', text)
        if quoted:
            quoted.sort(key=lambda item: (len(item.split()), len(item)))
            return quoted[0].strip()

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
            evidence_state=json.dumps(state, ensure_ascii=False, indent=2),
            round_id=round_id,
        )

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

    def _build_final_trace(
        self,
        pred: str,
        golden_answers: List[str],
        state: Dict[str, Any],
        review_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "pred": pred,
            "golden_answers": golden_answers,
            "answer_length": len(str(pred).split()),
            "used_forced_policy": state.get("decision") == "force_answer" or bool(state.get("failure_reason")),
            "failure_reason": state.get("failure_reason"),
            "slot_coverage": self._slot_coverage(state),
            "retrieved_more": any(
                entry.get("retrieval_plan") or entry.get("new_doc_count", 0)
                for entry in review_history or []
            ),
        }

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
            policy = self._build_adoption_policy(state, docs, failure_reason=failure_reason)
            policy["policy_parse_error"] = error
            return policy

        fallback = self._build_adoption_policy(state, docs, failure_reason=failure_reason)
        policy = {
            "trusted_claims": parsed.get("trusted_claims", fallback["trusted_claims"]),
            "answer_candidates": parsed.get("answer_candidates", fallback["answer_candidates"]),
            "ignored_docs": parsed.get("ignored_docs", fallback["ignored_docs"]),
            "conflict_resolution": parsed.get("conflict_resolution", fallback["conflict_resolution"]),
            "must_use_slots": parsed.get("must_use_slots", fallback["must_use_slots"]),
            "insufficient_slots": parsed.get("insufficient_slots", fallback["insufficient_slots"]),
            "answer_style": parsed.get("answer_style", "short_answer"),
            "failure_reason": parsed.get("failure_reason", failure_reason),
        }
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

        for iteration in range(self.max_iterations):
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
            coverage = self._slot_coverage(evidence_state)
            if (
                evidence_state.get("decision") == "continue_retrieval"
                and self._should_short_circuit_single_hop(evidence_state, coverage)
            ):
                evidence_state["decision"] = "ready_to_answer"
                evidence_state["retrieval_plan"] = []
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
                adoption_policy = self._build_adoption_policy(evidence_state, current_docs)
                break

            if evidence_state["decision"] in {"force_answer", "review_failed"}:
                forced_reason = evidence_state.get("failure_reason") or evidence_state["decision"]
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._finalize_policy_with_reviewer(
                    question, current_docs, evidence_state, review_history, forced_reason
                )
                break

            if new_claim_count == 0 and iteration > 0:
                forced_reason = "no_new_claim"
                evidence_state["decision"] = "force_answer"
                evidence_state["failure_reason"] = forced_reason
                current_docs, rank_trace = self._rank_docs_by_state(current_docs, evidence_state, return_trace=True)
                rank_trace["iteration"] = iteration
                diagnostic_trace["ranking_trace"].append(rank_trace)
                adoption_policy = self._finalize_policy_with_reviewer(
                    question, current_docs, evidence_state, review_history, forced_reason
                )
                review_history[-1]["decision"] = "force_answer"
                review_history[-1]["failure_reason"] = forced_reason
                diagnostic_trace["review_trace"][-1]["decision_after_merge"] = "force_answer"
                diagnostic_trace["review_trace"][-1]["failure_reason"] = forced_reason
                break

            retrieval_plan = evidence_state.get("retrieval_plan", [])
            if retrieval_plan:
                new_docs, retrieval_trace = self._retrieve_for_plans(
                    retrieval_plan,
                    iteration=iteration,
                    existing_docs=current_docs,
                    return_trace=True,
                )
                diagnostic_trace["retrieval_trace"].extend(retrieval_trace)
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
                adoption_policy = self._finalize_policy_with_reviewer(
                    question, current_docs, evidence_state, review_history, forced_reason
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
            adoption_policy = self._finalize_policy_with_reviewer(
                question, current_docs, evidence_state, review_history, forced_reason
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

        # Store intermediate results
        item.update_output("retrieval_result", final_docs)
        item.update_output("adoption_strategy", adoption_strategy)
        item.update_output("adoption_policy", adoption_policy)
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

        # Generate final answer
        pred = self.generator.generate([final_prompt])[0]
        pred = self._clean_final_answer(pred, adoption_policy, evidence_state)
        item.update_output("pred", pred)
        diagnostic_trace["final_trace"] = self._build_final_trace(
            pred,
            item.golden_answers,
            evidence_state,
            review_history,
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

        for iteration in range(self.max_iterations):
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices:
                print(f"  All items completed at iteration {iteration}")
                break

            n_active = len(active_indices)
            n_completed = n_items - n_active
            print(f"\n  Iter {iteration + 1}/{self.max_iterations}: {n_active} active, {n_completed} done")

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
                coverage = self._slot_coverage(evidence_states[idx])
                if (
                    evidence_states[idx].get("decision") == "continue_retrieval"
                    and self._should_short_circuit_single_hop(evidence_states[idx], coverage)
                ):
                    evidence_states[idx]["decision"] = "ready_to_answer"
                    evidence_states[idx]["retrieval_plan"] = []
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
                    adoption_policies[idx] = self._build_adoption_policy(evidence_states[idx], current_docs[idx])
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
                    )
                    adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])
                    active_mask[idx] = False
                    n_forced += 1
                    if decision == "review_failed":
                        n_failed += 1
                elif new_claim_count == 0 and iteration > 0:
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
                    if retrieval_plan:
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
                        diagnostic_traces[owner_idx]["retrieval_trace"].append({
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

                    for idx, new_docs in docs_by_idx.items():
                        before_ids = {doc.get("id") for doc in current_docs[idx]}
                        current_docs[idx] = self.merge_and_filter_docs(current_docs[idx], new_docs, [])
                        after_ids = {doc.get("id") for doc in current_docs[idx]}
                        review_histories[idx][-1]["new_doc_count"] = len(after_ids - before_ids)
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
                )
            if adoption_strategies[idx] is None:
                adoption_strategies[idx] = self._render_adoption_policy(adoption_policies[idx])

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

        pred_answers = self._batched_generate(self.generator, all_prompts)
        pred_answers = [
            self._clean_final_answer(pred, adoption_policies[idx], evidence_states[idx])
            for idx, pred in enumerate(pred_answers)
        ]
        dataset.update_output("pred", pred_answers)
        for idx, pred in enumerate(pred_answers):
            diagnostic_traces[idx]["final_trace"] = self._build_final_trace(
                pred,
                dataset.data[idx].golden_answers,
                evidence_states[idx],
                review_histories[idx],
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
