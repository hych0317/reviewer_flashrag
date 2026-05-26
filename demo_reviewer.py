"""
Demo script for ReviewerPipeline.

This script demonstrates the ReviewerPipeline workflow using real FlashRAG components
(retriever, generator) with detailed output of each step's content.

Usage:
    python demo_reviewer.py --model_path /path/to/model --retriever_path /path/to/retriever

Example:
    python demo_reviewer.py --num_samples 2 --model_path "/home/flyvideo/Sata2/hyc/models/Qwen3-4B-Instruct" --retriever_path "/home/flyvideo/Sata2/hyc/models/Retriever/e5-base-v2"
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ============================================================================
# Verbose ReviewerPipeline - extends ReviewerPipeline with detailed output
# ============================================================================

class VerboseReviewerPipeline:
    """
    A wrapper that runs ReviewerPipeline with verbose output for demonstration.
    Shows all intermediate steps including:
    - Initial retrieval results
    - Each review iteration (prompt, response, decision)
    - Document merging and filtering
    - Final generation with adoption strategy
    """
    def __init__(self, config, retriever, generator, log_fp, max_iterations=3, max_docs=10):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = max_iterations
        self.max_docs = max_docs
        self.log_fp = log_fp

        # Import prompts from reviewer_pipeline
        from reviewer_pipeline import (
            REVIEWER_SYSTEM_PROMPT,
            REVIEWER_USER_PROMPT,
            FINAL_GENERATION_SYSTEM_PROMPT,
            FINAL_GENERATION_USER_PROMPT,
            FORCED_ADOPTION_STRATEGY_PROMPT
        )
        self.REVIEWER_SYSTEM_PROMPT = REVIEWER_SYSTEM_PROMPT
        self.REVIEWER_USER_PROMPT = REVIEWER_USER_PROMPT
        self.FINAL_GENERATION_SYSTEM_PROMPT = FINAL_GENERATION_SYSTEM_PROMPT
        self.FINAL_GENERATION_USER_PROMPT = FINAL_GENERATION_USER_PROMPT
        self.FORCED_ADOPTION_STRATEGY_PROMPT = FORCED_ADOPTION_STRATEGY_PROMPT

        from flashrag.prompt import PromptTemplate
        self.reviewer_prompt_template = PromptTemplate(
            config=config,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=REVIEWER_USER_PROMPT,
        )
        self.generation_prompt_template = PromptTemplate(
            config=config,
            system_prompt=FINAL_GENERATION_SYSTEM_PROMPT,
            user_prompt=FINAL_GENERATION_USER_PROMPT,
        )
        self.forced_adoption_prompt_template = PromptTemplate(
            config=config,
            system_prompt="You are a document reviewer that must now provide a final adoption strategy.",
            user_prompt=FORCED_ADOPTION_STRATEGY_PROMPT,
        )

    def log(self, text: str = ""):
        self.log_fp.write(text + "\n")
        self.log_fp.flush()

    def print_separator(self, title: str, char: str = "=", width: int = 80):
        self.log(f"\n{char * width}")
        self.log(f" {title}")
        self.log(f"{char * width}\n")

    def print_step(self, step_num: int, title: str):
        self.log(f"\n{'─' * 60}")
        self.log(f"Step {step_num}: {title}")
        self.log(f"{'─' * 60}\n")

    def print_docs(self, docs: List[Dict], title: str = "Documents"):
        self.log(f"\n{title} ({len(docs)} documents):")
        for i, doc in enumerate(docs):
            content = doc.get('contents', doc.get('content', 'N/A'))
            self.log(f"  [{i}] ID: {doc.get('id', 'N/A')}")
            self.log(f"      Content: {content}\n")

    def format_docs_for_review(self, docs: List[Dict]) -> str:
        formatted = []
        for idx, doc in enumerate(docs):
            content = doc.get("contents", doc.get("content", ""))
            doc_id = doc.get("id", f"doc_{idx}")
            formatted.append(f"[Document {idx}] (ID: {doc_id})\n{content}\n")
        return "\n".join(formatted)

    def format_docs_for_generation(self, docs: List[Dict]) -> str:
        formatted = []
        for idx, doc in enumerate(docs):
            content = doc.get("contents", doc.get("content", ""))
            formatted.append(f"Document {idx + 1}:\n{content}\n")
        return "\n".join(formatted)

    def parse_reviewer_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            parsed = json.loads(json_str)
            parsed.setdefault("decision", "READY_TO_ANSWER")
            parsed.setdefault("evaluation", {})
            parsed.setdefault("explanation", "")
            parsed.setdefault("new_query", None)
            parsed.setdefault("docs_to_remove", [])
            parsed.setdefault("adoption_strategy", "Trust all provided documents.")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [Warning] Failed to parse reviewer response: {e}")
            return {
                "decision": "READY_TO_ANSWER",
                "evaluation": {},
                "explanation": f"Parse error: {e}",
                "new_query": None,
                "docs_to_remove": [],
                "adoption_strategy": "Trust all provided documents."
            }

    def merge_and_filter_docs(self, old_docs, new_docs, docs_to_remove):
        filtered_old = [doc for idx, doc in enumerate(old_docs) if idx not in docs_to_remove]
        existing_ids = {doc.get("id") for doc in filtered_old}
        merged = filtered_old.copy()
        for doc in new_docs:
            doc_id = doc.get("id")
            if doc_id not in existing_ids:
                merged.append(doc)
                existing_ids.add(doc_id)
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

    def run_single(self, question: str, golden_answers: List[str] = None):
        """Run the pipeline for a single question with verbose output."""
        self.print_separator(f"Question: {question}", "=")
        if golden_answers:
            self.log(f"Golden Answer(s): {golden_answers}\n")

        # ===== Step 1: Initial Retrieval =====
        self.print_step(1, "Initial Retrieval")
        self.log(f"Query: {question}")

        initial_docs = self.retriever.search(question)
        self.print_docs(initial_docs, "Retrieved Documents")

        # ===== Step 2: Review Loop =====
        self.print_step(2, "Review Loop")

        current_docs = initial_docs
        adoption_strategy = None  # Will be set by reviewer, not hardcoded
        review_history = []
        loop_completed_normally = True  # Track if we exited via break

        for iteration in range(self.max_iterations):
            self.log(f"\n{'='*40}")
            self.log(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.log(f"{'='*40}")

            # Format documents for review
            formatted_docs = self.format_docs_for_review(current_docs)

            # Create reviewer prompt
            reviewer_prompt = self.reviewer_prompt_template.get_string(
                question=question,
                formatted_docs=formatted_docs
            )

            self.log(f"\n[Reviewer Prompt] (length: {len(reviewer_prompt)} chars)")
            self.log("-" * 40)
            truncated = reviewer_prompt[:1000] + "..." if len(reviewer_prompt) > 1000 else reviewer_prompt
            self.log(truncated)

            # Get reviewer response
            self.log(f"\n[Calling Reviewer LLM...]")
            reviewer_response = self.generator.generate([reviewer_prompt])[0]

            self.log(f"\n[Reviewer Response]")
            self.log("-" * 40)
            self.log(reviewer_response)

            # Parse response
            parsed = self.parse_reviewer_response(reviewer_response)

            self.log(f"\n[Parsed Decision]")
            self.log(f"  Decision: {parsed.get('decision')}")
            self.log(f"  Evaluation: {parsed.get('evaluation')}")
            self.log(f"  Explanation: {parsed.get('explanation')}")

            review_history.append({
                "iteration": iteration,
                "num_docs": len(current_docs),
                "decision": parsed["decision"],
                "evaluation": parsed["evaluation"],
                "explanation": parsed["explanation"],
            })

            if parsed["decision"] == "READY_TO_ANSWER":
                adoption_strategy = parsed.get("adoption_strategy")
                loop_completed_normally = False
                self.log(f"\n[Adoption Strategy Generated]")
                self.log(f"  {adoption_strategy}")
                break
            else:
                new_query = parsed.get("new_query")
                docs_to_remove = parsed.get("docs_to_remove", [])

                self.log(f"\n[Re-retrieval Required]")
                self.log(f"  New Query: {new_query}")
                self.log(f"  Documents to Remove: {docs_to_remove}")

                if new_query:
                    self.log(f"\n[Retrieving new documents...]")
                    new_docs = self.retriever.search(new_query)
                    self.print_docs(new_docs, "Newly Retrieved Documents")

                    current_docs = self.merge_and_filter_docs(current_docs, new_docs, docs_to_remove)
                    self.log(f"\n[After Merge] {len(current_docs)} documents")
                else:
                    current_docs = [doc for idx, doc in enumerate(current_docs) if idx not in docs_to_remove]
                    self.log(f"\n[After Filtering] {len(current_docs)} documents")

        # If max iterations reached without READY_TO_ANSWER, force adoption strategy
        if loop_completed_normally:
            self.log(f"\n{'='*40}")
            self.log(f"Max iterations reached - Forcing adoption strategy")
            self.log(f"{'='*40}")

            formatted_docs = self.format_docs_for_review(current_docs)
            review_summary = self._generate_review_summary(review_history)

            forced_prompt = self.forced_adoption_prompt_template.get_string(
                question=question,
                formatted_docs=formatted_docs,
                review_summary=review_summary
            )

            self.log(f"\n[Forced Adoption Prompt] (length: {len(forced_prompt)} chars)")
            self.log("-" * 40)
            truncated = forced_prompt[:1000] + "..." if len(forced_prompt) > 1000 else forced_prompt
            self.log(truncated)

            self.log(f"\n[Calling Reviewer LLM for forced adoption...]")
            forced_response = self.generator.generate([forced_prompt])[0]

            self.log(f"\n[Forced Adoption Response]")
            self.log("-" * 40)
            self.log(forced_response)

            parsed_forced = self.parse_reviewer_response(forced_response)
            adoption_strategy = parsed_forced.get("adoption_strategy")

            self.log(f"\n[Forced Adoption Strategy]")
            self.log(f"  {adoption_strategy}")

            review_history.append({
                "iteration": "forced",
                "num_docs": len(current_docs),
                "decision": "READY_TO_ANSWER (forced)",
                "evaluation": {},
                "explanation": "Max iterations reached, adoption strategy forced.",
            })

        # Fallback if adoption_strategy is still None (parse failure)
        if adoption_strategy is None:
            adoption_strategy = "The retrieved documents may not be reliable. Answer based on general knowledge if possible, and acknowledge uncertainty."
            self.log(f"\n[Using fallback adoption strategy due to parse failure]")
            self.log(f"  {adoption_strategy}")

        # ===== Step 3: Final Generation =====
        self.print_step(3, "Final Generation")

        formatted_reference = self.format_docs_for_generation(current_docs)

        final_prompt = self.generation_prompt_template.get_string(
            question=question,
            reference=formatted_reference,
            adoption_strategy=adoption_strategy
        )

        self.log(f"[Final Generation Prompt] (length: {len(final_prompt)} chars)")
        self.log("-" * 40)
        truncated = final_prompt[:1200] + "..." if len(final_prompt) > 1200 else final_prompt
        self.log(truncated)

        self.log(f"\n[Calling Generator LLM...]")
        final_answer = self.generator.generate([final_prompt])[0]

        self.log(f"\n[Final Answer]")
        self.log("-" * 40)
        self.log(f"  Predicted: {final_answer}")
        if golden_answers:
            self.log(f"  Golden: {golden_answers}")
            is_correct = any(ga.lower() in final_answer.lower() for ga in golden_answers)
            self.log(f"\n  Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return {
            "question": question,
            "golden_answers": golden_answers,
            "predicted": final_answer,
            "adoption_strategy": adoption_strategy,
            "review_history": review_history,
            "final_docs": current_docs,
        }


def main():
    parser = argparse.ArgumentParser(description="ReviewerPipeline Demo with Real Components")

    # Required arguments
    parser.add_argument("--model_path", type=str, default="/home/flyvideo/Sata2/hyc/models/Qwen3-4B-Instruct",
                       help="Path to generator model")
    parser.add_argument("--retriever_path", type=str, default="/home/flyvideo/Sata2/hyc/models/Retriever/e5-base-v2",
                       help="Path to retriever model")

    # Optional arguments
    parser.add_argument("--dataset_path", type=str, default="dataset/nq/test.jsonl",
                       help="Path to dataset")
    parser.add_argument("--index_path", type=str, default="/home/flyvideo/Sata2/hyc/projects/RAG/FlashRAG/data/wiki_index/e5_Flat.index",
                       help="Path to FAISS index")
    parser.add_argument("--corpus_path", type=str, default="/home/flyvideo/Sata2/hyc/projects/RAG/FlashRAG/data/wiki18_100w.jsonl",
                       help="Path to corpus")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples to process")
    parser.add_argument("--framework", type=str, default="hf",
                       choices=["hf", "vllm", "fschat"],
                       help="Framework for generation")
    parser.add_argument("--gpu_id", type=str, default="0,2",
                       help="GPU ID")
    parser.add_argument("--max_iterations", type=int, default=3,
                       help="Max review iterations")
    parser.add_argument("--max_docs", type=int, default=10,
                       help="Max documents to keep")
    parser.add_argument("--retrieval_topk", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--log_path", type=str, default="reviewer_log.txt",
                       help="Path to save verbose log output")

    args = parser.parse_args()

    # Resolve paths relative to quick_start directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    quick_start_dir = os.path.join(script_dir, "../quick_start")

    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.join(quick_start_dir, args.dataset_path)
    if not os.path.isabs(args.index_path):
        args.index_path = os.path.join(quick_start_dir, args.index_path)
    if not os.path.isabs(args.corpus_path):
        args.corpus_path = os.path.join(quick_start_dir, args.corpus_path)

    # Validate paths
    for name, path in [("Dataset", args.dataset_path), ("Index", args.index_path), ("Corpus", args.corpus_path)]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            sys.exit(1)

    with open(args.log_path, "w", encoding="utf-8") as log_fp:
        def log(msg: str = ""):
            log_fp.write(msg + "\n")
            log_fp.flush()

        log("=" * 80)
        log(" ReviewerPipeline Demo - Real Components")
        log("=" * 80)
        log(f"\nConfiguration:")
        log(f"  Model: {args.model_path}")
        log(f"  Retriever: {args.retriever_path}")
        log(f"  Index: {args.index_path}")
        log(f"  Corpus: {args.corpus_path}")
        log(f"  Framework: {args.framework}")
        log(f"  GPU: {args.gpu_id}")
        log(f"  Max Iterations: {args.max_iterations}")
        log(f"  Retrieval Top-K: {args.retrieval_topk}")
        log(f"  Log Path: {args.log_path}")

        # Setup FlashRAG config
        from flashrag.config import Config
        from flashrag.utils import get_retriever, get_generator

        config_dict = {
            "index_path": args.index_path,
            "corpus_path": args.corpus_path,
            "model2path": {
                "e5": args.retriever_path,
                "generator": args.model_path,
            },
            "generator_model": "generator",
            "generator_model_path": args.model_path,
            "retrieval_method": "e5",
            "framework": args.framework,
            "retrieval_topk": args.retrieval_topk,
            "gpu_id": args.gpu_id,
            "generation_params": {
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        }

        log(f"\n[Initializing FlashRAG components...]")
        config = Config(config_dict=config_dict)

        log(f"[Loading retriever...]")
        retriever = get_retriever(config)

        log(f"[Loading generator...]")
        generator = get_generator(config)

        # Create verbose pipeline
        pipeline = VerboseReviewerPipeline(
            config=config,
            retriever=retriever,
            generator=generator,
            log_fp=log_fp,
            max_iterations=args.max_iterations,
            max_docs=args.max_docs,
        )

        # Load dataset
        log(f"\n[Loading dataset from {args.dataset_path}...]")
        samples = []
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= args.num_samples:
                    break
                samples.append(json.loads(line))

        log(f"[Loaded {len(samples)} samples]")

        # Run pipeline
        results = []
        for i, sample in enumerate(samples):
            log(f"\n\n{'#' * 80}")
            log(f" SAMPLE {i + 1}/{len(samples)}")
            log(f"{'#' * 80}")
            result = pipeline.run_single(
                question=sample["question"],
                golden_answers=sample.get("golden_answers", [])
            )
            results.append(result)

        # Summary
        log(f"\n\n{'=' * 80}")
        log(" SUMMARY")
        log("=" * 80)
        correct = 0
        for i, result in enumerate(results):
            golden = result.get("golden_answers", [])
            pred = result.get("predicted", "")
            is_correct = any(ga.lower() in pred.lower() for ga in golden) if golden else None

            log(f"\n[{i+1}] Q: {result['question']}")
            log(f"    Predicted: {pred[:100]}{'...' if len(pred) > 100 else ''}")
            log(f"    Golden: {golden}")
            if is_correct is not None:
                log(f"    Correct: {is_correct}")
                if is_correct:
                    correct += 1

        if results and results[0].get("golden_answers"):
            log(f"\nAccuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        log("\n[Demo Complete]")


if __name__ == "__main__":
    main()
