"""
Running script for the Reviewer Pipeline.

This script provides different configurations for running the ReviewerPipeline,
including options for using a separate reviewer model (e.g., Qwen3) from the
generator model.

Usage:
    python run_reviewer.py --method_name reviewer --dataset_name nq --split test --gpu_id 0

    # With separate reviewer model
    python run_reviewer.py --method_name reviewer_separate --dataset_name hotpotqa --split dev --gpu_id 0,1
"""

import os
import sys
import argparse

# Ensure the project root is in path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from flashrag.config import Config
from flashrag.utils import get_dataset


def build_save_note(base_note, result_tag):
    if result_tag:
        return f"{base_note}-{result_tag}"
    return base_note


def build_save_dir(dataset_name, split):
    result_split = "dev" if split == "test" else split
    return os.path.join("./result", result_split, dataset_name)


def reviewer_reasoning(args):
    """
    Run the ReviewerPipeline with a reasoning-optimized model.

    Uses models that are better at structured reasoning and evaluation,
    such as models trained with reasoning capabilities.
    """
    save_note = build_save_note("reviewer-reasoning", args.result_tag)
    config_dict = {
        "save_note": save_note,
        "save_dir": build_save_dir(args.dataset_name, args.split),
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "framework": "vllm",
        "generator_max_input_len": 16384,
        "generation_params": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "skip_special_tokens": True,
        },
    }
    if args.generator_model:
        config_dict["generator_model"] = args.generator_model

    config = Config("reviewer_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from reviewer_pipeline import ReviewerPipeline

    pipeline = ReviewerPipeline(
        config,
        max_iterations=args.max_iter,
        max_docs=args.max_docs,
    )
    result = pipeline.run(test_data, sample_size=args.sample_size)

    return result


def reviewer_multihop(args):
    """
    Run the ReviewerPipeline optimized for multi-hop QA tasks.

    This configuration uses more iterations and keeps more documents
    for complex multi-hop reasoning tasks like HotpotQA, 2WikiMultiHopQA.
    """
    save_note = build_save_note("reviewer-multihop", args.result_tag)
    config_dict = {
        "save_note": save_note,
        "save_dir": build_save_dir(args.dataset_name, args.split),
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "framework": "vllm",
        "retrieval_topk": 10,  # Retrieve more docs initially
        "generator_max_input_len": 8192,
        "generation_params": {
            "max_tokens": 512,
            "temperature": 0.0,
        },
    }
    if args.generator_model:
        config_dict["generator_model"] = args.generator_model

    config = Config("reviewer_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from reviewer_pipeline import ReviewerPipeline

    # More iterations and docs for multi-hop
    pipeline = ReviewerPipeline(
        config,
        max_iterations=max(args.max_iter, 3),  # At least 5 iterations for multi-hop
        max_docs=max(args.max_docs, 10),  # Keep more documents
    )
    result = pipeline.run(test_data, sample_size=args.sample_size)

    return result


# Method registry
METHOD_REGISTRY = {
    "reviewer": reviewer_multihop,
    "reviewer_reasoning": reviewer_reasoning,
    "reviewer_multihop": reviewer_multihop,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Reviewer Pipeline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reviewer pipeline
  python run_reviewer.py --method_name reviewer --dataset_name nq --split test --gpu_id 0

  # Reviewer with separate Qwen3 model
  python run_reviewer.py --method_name reviewer_qwen --dataset_name hotpotqa --split dev --gpu_id 0

  # Multi-hop optimized
  python run_reviewer.py --method_name reviewer_multihop --dataset_name 2wikimultihopqa --split dev --gpu_id 0,1
        """
    )

    # Required arguments
    parser.add_argument("--method_name", type=str, required=True, choices=list(METHOD_REGISTRY.keys()),
                        help="Name of the reviewer method to run")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., nq, triviaqa, hotpotqa)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (train/dev/test)")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID(s) to use, comma-separated for multiple GPUs")

    # Reviewer-specific arguments
    parser.add_argument("--max_iter", type=int, default=3,
                        help="Maximum number of review iterations (default: 3)")
    parser.add_argument("--max_docs", type=int, default=10,
                        help="Maximum number of documents to keep (default: 10)")
    parser.add_argument("--reviewer_model_path", type=str, default=None,
                        help="Path to reviewer model (for reviewer_separate method)")
    parser.add_argument("--generator_model", type=str, default=None,
                        help="Generator model name/path to override reviewer_config.yaml")
    parser.add_argument("--sample_size", type=int, default=3000,
                        help="Number of samples to run; use <=0 for the full split (default: 3000)")
    parser.add_argument("--result_tag", type=str, default="",
                        help="Optional suffix appended to save_note for distinguishing result directories")

    args = parser.parse_args()

    # Get and run the method
    if args.method_name not in METHOD_REGISTRY:
        print(f"Unknown method: {args.method_name}")
        print(f"Available methods: {list(METHOD_REGISTRY.keys())}")
        sys.exit(1)

    method_func = METHOD_REGISTRY[args.method_name]
    print(f"\n{'='*60}")
    print(f"Running: {args.method_name}")
    print(f"Dataset: {args.dataset_name} ({args.split})")
    print(f"GPU: {args.gpu_id}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Max docs: {args.max_docs}")
    print(f"Generator model: {args.generator_model or 'config default'}")
    print(f"Sample size: {args.sample_size}")
    print(f"Result tag: {args.result_tag or 'none'}")
    print(f"{'='*60}\n")

    result = method_func(args)


if __name__ == "__main__":
    main()
