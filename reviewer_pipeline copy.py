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

import re
import json
from typing import List, Dict, Any, Tuple, Optional

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate


# ============================================================================
# Reviewer Prompt Templates
# ============================================================================

REVIEWER_SYSTEM_PROMPT = """You are a critical document reviewer.
Your task is to evaluate whether the retrieved documents are sufficient and reliable to answer the given question.\n

You must evaluate the documents based on the following dimensions:
1. **Information Sufficiency**: Do the documents contain enough information to answer the question completely?
2. **Factual Consistency**: Are there any contradictions or conflicts between documents?
3. **Document Credibility**: Are the documents reliable and trustworthy? Do they contain harmful, misleading, or incorrect information?

Based on your evaluation, you must make ONE of two decisions:

**Decision A - NEED_MORE_INFO**: If the documents are insufficient, conflicting, or unreliable:
- Provide a brief explanation of what is missing or problematic
- Generate a NEW, more precise search query to retrieve better documents
- Identify any documents that should be REMOVED due to harmful/incorrect information

**Decision B - READY_TO_ANSWER**: If the documents are sufficient and reliable:
- Generate a concise "Adoption Strategy" (2-3 sentences) to guide the generator
- Briefly state: which documents to trust/ignore, how to handle conflicts, and the recommended approach
- Example: "Documents 0 and 2 are relevant. Document 1 contains outdated info, ignore it. Synthesize the answer from Doc 0 and 2."

You MUST respond in the following JSON format:
```json
{{
    "decision": "NEED_MORE_INFO" or "READY_TO_ANSWER",
    "evaluation": {{
        "sufficiency": "sufficient/partial/insufficient",
        "conflicts": "none/minor/major",
        "credibility": "high/medium/low",
        "problematic_docs": [list of document indices with issues, e.g., [0, 2]]
    }},
    "explanation": "Brief explanation of your evaluation",
    "new_query": "New search query (only if decision is NEED_MORE_INFO, otherwise null)",
    "docs_to_remove": [list of document indices to remove, e.g., [1, 3]],
    "adoption_strategy": "Meta-cognitive adoption strategy (only if decision is READY_TO_ANSWER, otherwise null)"
}}
```"""

REVIEWER_USER_PROMPT = """Question: {question}

Retrieved Documents:
{formatted_docs}

Please evaluate these documents and provide your decision in the specified JSON format."""


FORCED_ADOPTION_STRATEGY_PROMPT = """You have reached the maximum number of review iterations for this question.
Based on your previous evaluations, you MUST now generate a final adoption strategy.

Question: {question}

Current Documents:
{formatted_docs}

Previous Review Summary:
{review_summary}

Based on the review history, generate a concise adoption strategy (2-3 sentences max):

**If documents are COMPLETELY IRRELEVANT or HARMFUL**:
- Set `docs_to_remove` to ALL indices, adoption_strategy: "Ignore all documents. Answer from general knowledge."

**If documents have LIMITED RELEVANCE**:
- Keep documents, adoption_strategy: "Documents have limited relevance. Only use directly relevant info, verify with your knowledge."

**If documents are SUFFICIENT**:
- adoption_strategy: Briefly specify which docs to trust and how to synthesize.

Respond in JSON format:
```json
{{
    "decision": "READY_TO_ANSWER",
    "docs_to_remove": [list of document indices to remove, or empty if keeping all],
    "adoption_strategy": "Your final adoption strategy here"
}}
```"""

FINAL_GENERATION_SYSTEM_PROMPT = """You are a concise assistant. Your task is to answer questions strictly based on the provided documents and the adoption strategy below.

Adoption Strategy:
{adoption_strategy}

Using the strategy above, answer the question provided by the user.

Output requirements:
- Only output the final answer.
- Do not include any explanations, prefixes, or additional text.

Expected output format examples:
- Question: who got the first nobel prize in physics  
Answer: Wilhelm Conrad Röntgen
- Question: when was obama born
Answer: August 4, 1961
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
        self.batch_size = batch_size  # Max prompts per generate call

        # Create reviewer prompt template
        self.reviewer_prompt_template = PromptTemplate(
            config=config,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=REVIEWER_USER_PROMPT,
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

    def parse_reviewer_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the reviewer LLM."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            parsed = json.loads(json_str)

            # Validate required fields
            if "decision" not in parsed:
                parsed["decision"] = "READY_TO_ANSWER"  # Default to ready

            # Ensure default values
            parsed.setdefault("evaluation", {})
            parsed.setdefault("explanation", "")
            parsed.setdefault("new_query", None)
            parsed.setdefault("docs_to_remove", [])
            parsed.setdefault("adoption_strategy", "Trust all provided documents and synthesize the information to answer the question.")

            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            # If parsing fails, return a default "ready" decision
            return {
                "decision": "READY_TO_ANSWER",
                "evaluation": {"sufficiency": "unknown", "conflicts": "unknown", "credibility": "unknown"},
                "explanation": f"Failed to parse reviewer response: {str(e)}",
                "new_query": None,
                "docs_to_remove": [],
                "adoption_strategy": "Trust all provided documents and synthesize the information to answer the question."
            }

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
            summary_parts.append(
                f"Iteration {entry['iteration'] + 1}: "
                f"Decision={entry['decision']}, "
                f"Sufficiency={eval_info.get('sufficiency', 'unknown')}, "
                f"Credibility={eval_info.get('credibility', 'unknown')}. "
                f"Explanation: {entry.get('explanation', 'N/A')}"
            )
        return "\n".join(summary_parts)

    def run_review_loop(
        self,
        question: str,
        initial_docs: List[Dict]
    ) -> Tuple[List[Dict], str, List[Dict]]:
        """
        Execute the review loop for a single question.

        Args:
            question: The input question
            initial_docs: Initially retrieved documents

        Returns:
            Tuple of (final_docs, adoption_strategy, review_history)
        """
        current_docs = initial_docs
        review_history = []
        adoption_strategy = None  # Will be set by reviewer, not hardcoded
        loop_completed_normally = True  # Track if we exited via break

        for iteration in range(self.max_iterations):
            # Format documents for review
            formatted_docs = self.format_docs_for_review(current_docs)

            # Create reviewer prompt
            reviewer_prompt = self.reviewer_prompt_template.get_string(
                question=question,
                formatted_docs=formatted_docs
            )

            # Get reviewer evaluation
            reviewer_response = self.reviewer.generate([reviewer_prompt])[0]

            # Parse response
            parsed_response = self.parse_reviewer_response(reviewer_response)

            # Record history
            review_history.append({
                "iteration": iteration,
                "num_docs": len(current_docs),
                "decision": parsed_response["decision"],
                "evaluation": parsed_response["evaluation"],
                "explanation": parsed_response["explanation"],
            })

            if parsed_response["decision"] == "READY_TO_ANSWER":
                # Exit loop with adoption strategy
                adoption_strategy = parsed_response.get("adoption_strategy")
                loop_completed_normally = False
                break

            else:  # NEED_MORE_INFO
                # Get new query
                new_query = parsed_response.get("new_query")
                docs_to_remove = parsed_response.get("docs_to_remove", [])

                if new_query:
                    # Retrieve new documents
                    new_docs = self.retriever.search(new_query)

                    # Merge and filter
                    current_docs = self.merge_and_filter_docs(
                        current_docs,
                        new_docs,
                        docs_to_remove
                    )
                else:
                    # Just remove problematic docs if no new query
                    current_docs = [
                        doc for idx, doc in enumerate(current_docs)
                        if idx not in docs_to_remove
                    ]

        # If max iterations reached without READY_TO_ANSWER, force adoption strategy generation
        if loop_completed_normally:
            formatted_docs = self.format_docs_for_review(current_docs)
            review_summary = self._generate_review_summary(review_history)

            forced_prompt = self.forced_adoption_prompt_template.get_string(
                question=question,
                formatted_docs=formatted_docs,
                review_summary=review_summary
            )

            forced_response = self.reviewer.generate([forced_prompt])[0]
            parsed_forced = self.parse_reviewer_response(forced_response)
            adoption_strategy = parsed_forced.get("adoption_strategy")

            # Handle docs_to_remove from forced response (Case A: completely irrelevant docs)
            forced_docs_to_remove = parsed_forced.get("docs_to_remove", [])
            if forced_docs_to_remove:
                current_docs = [
                    doc for idx, doc in enumerate(current_docs)
                    if idx not in forced_docs_to_remove
                ]

            # Record forced adoption in history
            review_history.append({
                "iteration": "forced",
                "num_docs": len(current_docs),
                "decision": "READY_TO_ANSWER (forced)",
                "evaluation": {},
                "explanation": f"Max iterations reached. Docs removed: {forced_docs_to_remove}",
            })

        # Fallback if adoption_strategy is still None (parse failure)
        if adoption_strategy is None:
            adoption_strategy = "The retrieved documents may not be reliable. Answer based on general knowledge if possible, and acknowledge uncertainty."

        return current_docs, adoption_strategy, review_history

    def run_item(self, item) -> None:
        """Process a single item through the reviewer pipeline."""
        question = item.question

        # Step 1: Initial retrieval
        initial_docs = self.retriever.search(question)
        item.update_output("initial_retrieval_result", initial_docs)

        # Step 2: Review loop
        final_docs, adoption_strategy, review_history = self.run_review_loop(
            question, initial_docs
        )

        # Store intermediate results
        item.update_output("retrieval_result", final_docs)
        item.update_output("adoption_strategy", adoption_strategy)
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
        item.update_output("pred", pred)

    def run_batch(self, dataset) -> None:
        """
        Process a batch of items through the reviewer pipeline using layered batch processing.

        Instead of processing items one-by-one (which causes N * max_iterations generate calls),
        this method processes all items layer-by-layer, enabling efficient batch inference.

        Flow:
        - Iteration 1: All N items → batch review → split into (done, need_more)
        - Iteration 2: Remaining M items → batch review → ...
        - ... until all done or max_iterations reached
        - Final: Batch generate all answers
        """
        questions = dataset.question
        n_items = len(questions)

        print(f"\n{'='*60}")
        print(f"Reviewer Pipeline: {n_items} items, max {self.max_iterations} iterations")
        print(f"{'='*60}")

        # Step 1: Batch initial retrieval
        print(f"\n[1/4] Initial retrieval...")
        initial_retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("initial_retrieval_result", initial_retrieval_results)
        print(f"  Done: retrieved docs for {n_items} questions")

        # Initialize state tracking for all items
        current_docs = [docs.copy() if docs else [] for docs in initial_retrieval_results]
        adoption_strategies = [None] * n_items
        review_histories = [[] for _ in range(n_items)]
        active_mask = [True] * n_items  # True = still needs processing

        # Step 2: Layered batch review loop
        print(f"\n[2/4] Review loop...")

        for iteration in range(self.max_iterations):
            # Find items that still need processing
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices:
                print(f"  All items completed at iteration {iteration}")
                break

            n_active = len(active_indices)
            n_completed = n_items - n_active
            print(f"\n  Iter {iteration + 1}/{self.max_iterations}: {n_active} active, {n_completed} done")

            # Batch prepare reviewer prompts for active items
            reviewer_prompts = []
            for idx in active_indices:
                formatted_docs = self.format_docs_for_review(current_docs[idx])
                prompt = self.reviewer_prompt_template.get_string(
                    question=questions[idx],
                    formatted_docs=formatted_docs
                )
                reviewer_prompts.append(prompt)

            # Batch call reviewer (with internal batching for memory control)
            print(f"    Reviewing {n_active} items...")
            reviewer_responses = self._batched_generate(self.reviewer, reviewer_prompts)

            # Process responses and track which items need re-retrieval
            need_retrieval_indices = []
            new_queries = []
            n_ready = 0
            n_need_more = 0

            for batch_idx, idx in enumerate(active_indices):
                parsed = self.parse_reviewer_response(reviewer_responses[batch_idx])

                # Record history
                review_histories[idx].append({
                    "iteration": iteration,
                    "num_docs": len(current_docs[idx]),
                    "decision": parsed["decision"],
                    "evaluation": parsed["evaluation"],
                    "explanation": parsed["explanation"],
                })

                if parsed["decision"] == "READY_TO_ANSWER":
                    # Item is done
                    adoption_strategies[idx] = parsed.get("adoption_strategy")
                    # Handle any final doc removals
                    docs_to_remove = parsed.get("docs_to_remove", [])
                    if docs_to_remove:
                        current_docs[idx] = [
                            doc for i, doc in enumerate(current_docs[idx])
                            if i not in docs_to_remove
                        ]
                    active_mask[idx] = False
                    n_ready += 1
                else:
                    # NEED_MORE_INFO - prepare for re-retrieval
                    new_query = parsed.get("new_query")
                    docs_to_remove = parsed.get("docs_to_remove", [])

                    # Remove problematic docs first
                    if docs_to_remove:
                        current_docs[idx] = [
                            doc for i, doc in enumerate(current_docs[idx])
                            if i not in docs_to_remove
                        ]

                    if new_query:
                        need_retrieval_indices.append(idx)
                        new_queries.append(new_query)

                    n_need_more += 1

            print(f"    Results: {n_ready} ready, {n_need_more} need more")

            # Batch re-retrieve for items that need it
            if new_queries:
                print(f"    Re-retrieving for {len(new_queries)} items...")
                new_retrieval_results = self.retriever.batch_search(new_queries)

                for batch_idx, idx in enumerate(need_retrieval_indices):
                    current_docs[idx] = self.merge_and_filter_docs(
                        current_docs[idx],
                        new_retrieval_results[batch_idx],
                        []  # Already removed above
                    )

        # Step 3: Handle items that reached max_iterations without completing
        remaining_indices = [i for i, active in enumerate(active_mask) if active]
        if remaining_indices:
            print(f"\n[3/4] Forcing strategy for {len(remaining_indices)} remaining items...")

            # Batch prepare forced adoption prompts
            forced_prompts = []
            for idx in remaining_indices:
                formatted_docs = self.format_docs_for_review(current_docs[idx])
                review_summary = self._generate_review_summary(review_histories[idx])
                prompt = self.forced_adoption_prompt_template.get_string(
                    question=questions[idx],
                    formatted_docs=formatted_docs,
                    review_summary=review_summary
                )
                forced_prompts.append(prompt)

            # Batch call reviewer for forced adoption
            forced_responses = self._batched_generate(self.reviewer, forced_prompts)

            for batch_idx, idx in enumerate(remaining_indices):
                parsed = self.parse_reviewer_response(forced_responses[batch_idx])
                adoption_strategies[idx] = parsed.get("adoption_strategy")

                # Handle docs_to_remove from forced response
                forced_docs_to_remove = parsed.get("docs_to_remove", [])
                if forced_docs_to_remove:
                    current_docs[idx] = [
                        doc for i, doc in enumerate(current_docs[idx])
                        if i not in forced_docs_to_remove
                    ]

                # Record in history
                review_histories[idx].append({
                    "iteration": "forced",
                    "num_docs": len(current_docs[idx]),
                    "decision": "READY_TO_ANSWER (forced)",
                    "evaluation": {},
                    "explanation": f"Max iterations reached. Docs removed: {forced_docs_to_remove}",
                })

            print(f"  Done: {len(remaining_indices)} forced strategies")
        else:
            print(f"\n[3/4] All items completed within limit")

        # Fill in default adoption_strategy for any items that somehow don't have one
        for idx in range(n_items):
            if adoption_strategies[idx] is None:
                adoption_strategies[idx] = (
                    "The retrieved documents may not be fully reliable. "
                    "Answer based on the most relevant information and acknowledge uncertainty."
                )

        # Update dataset with intermediate results
        dataset.update_output("retrieval_result", current_docs)
        dataset.update_output("adoption_strategy", adoption_strategies)
        dataset.update_output("review_history", review_histories)

        # Step 4: Batch generate final answers
        print(f"\n[4/4] Generating {n_items} final answers...")

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
        dataset.update_output("pred", pred_answers)

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
