"""
Evaluation script for the Bluesky Post Explainer agent.

Runs the agent on all labeled test cases and scores the outputs using:
1. Embedding-based similarity: cosine similarity against the expected output.
2. LLM-as-a-judge: relevance, formatting, length, and citation metrics (each 0 or 1).

Usage:
    python evaluate.py [--verbose]

The results are printed as a table with per-case and aggregate scores.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_eval_module(name: str):
    """
    Load an eval/src module directly from its file path.

    Both eval/src/ and agent/src/ are named 'src', which causes a namespace
    conflict when both directories are on sys.path.  Loading eval's modules
    explicitly by file path avoids that conflict entirely.
    """
    path = Path(__file__).parent / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_eval_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_eval_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_sim_mod = _load_eval_module("similarity")
_judge_mod = _load_eval_module("llm_judge")

SimilarityScorer = _sim_mod.SimilarityScorer
LLMJudge = _judge_mod.LLMJudge
_JudgeResult = _judge_mod.JudgeResult

# Add agent/ to sys.path so 'src.pipeline' resolves to agent/src/pipeline.py
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from dotenv import load_dotenv
from src.pipeline import ExplainerPipeline

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth" / "test_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Bluesky Post Explainer agent against ground truth test cases.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show intermediate pipeline output for each test case",
    )
    return parser.parse_args()


def load_test_cases() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        print(f"No test cases found at {GROUND_TRUTH_PATH}")
        print("Add test cases with: python add_test_case.py --url <url> --expected '<output>'")
        sys.exit(0)

    with open(GROUND_TRUTH_PATH, "r") as f:
        cases = json.load(f)

    if not cases:
        print("No test cases found in test_cases.json.")
        print("Add test cases with: python add_test_case.py --url <url> --expected '<output>'")
        sys.exit(0)

    return cases


def print_header() -> None:
    header = f"{'ID':<8} {'Similarity':>10} {'Relevance':>10} {'Formatting':>11} {'Length':>7} {'Citation':>9} {'Average':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))


def print_row(case_id: str, sim: float, relevance: float, formatting: float, length: float, citation: float, avg: float) -> None:
    print(
        f"{case_id:<8} {sim:>10.3f} {relevance:>10.1f} {formatting:>11.1f} {length:>7.1f} {citation:>9.1f} {avg:>8.3f}"
    )


def print_footer(results: list[dict]) -> None:
    n = len(results)
    avg_sim = sum(r["similarity"] for r in results) / n
    avg_rel = sum(r["relevance"] for r in results) / n
    avg_fmt = sum(r["formatting"] for r in results) / n
    avg_len = sum(r["length"] for r in results) / n
    avg_cit = sum(r["citation"] for r in results) / n
    avg_all = sum(r["average"] for r in results) / n

    print("-" * 65)
    print_row("AVERAGE", avg_sim, avg_rel, avg_fmt, avg_len, avg_cit, avg_all)
    print("=" * 65)
    print(f"\nEvaluated {n} test case(s).")
    print(f"  Embedding similarity (vs ground truth):  {avg_sim:.3f}")
    print(f"  Relevance (LLM judge, 0-1):              {avg_rel:.3f}")
    print(f"  Formatting (LLM judge, 0-1):             {avg_fmt:.3f}")
    print(f"  Length (LLM judge, 0-1):                 {avg_len:.3f}")
    print(f"  Citation (LLM judge, 0-1):               {avg_cit:.3f}")
    print(f"  Overall average:                          {avg_all:.3f}\n")


def main() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    args = parse_args()
    test_cases = load_test_cases()

    print(f"Loaded {len(test_cases)} test case(s).")
    print("Initializing pipeline and evaluators...")

    pipeline = ExplainerPipeline(verbose=args.verbose)
    scorer = SimilarityScorer()
    judge = LLMJudge()

    results = []
    print_header()

    for case in test_cases:
        case_id = case["id"]
        url = case["url"]
        expected = case["expected_output"]

        print(f"\n[{case_id}] Running agent on: {url}")

        try:
            actual = pipeline.run(url)
        except Exception as exc:
            print(f"  ERROR running pipeline: {exc}")
            actual = ""

        if args.verbose:
            print(f"\n  Agent output:\n{actual}\n")
            print(f"  Expected:\n{expected}\n")

        # Metric 1: embedding similarity
        sim = scorer.score(actual, expected) if actual else 0.0

        # Metric 2: LLM-as-a-judge
        # Use the expected output as the "topic summary" so the judge
        # has context about what the post is about.
        if actual and actual != "Post did not pass profanity filter":
            judge_result = judge.evaluate(
                post_url=url,
                post_summary=expected,
                actual_output=actual,
            )
        else:
            judge_result = _JudgeResult(relevance=0.0, formatting=0.0, length=0.0, citation=0.0)

        avg = (sim + judge_result.average) / 2.0

        result = {
            "id": case_id,
            "similarity": sim,
            "relevance": judge_result.relevance,
            "formatting": judge_result.formatting,
            "length": judge_result.length,
            "citation": judge_result.citation,
            "average": avg,
        }
        results.append(result)
        print_row(case_id, sim, judge_result.relevance, judge_result.formatting, judge_result.length, judge_result.citation, avg)

    print_footer(results)


if __name__ == "__main__":
    main()
