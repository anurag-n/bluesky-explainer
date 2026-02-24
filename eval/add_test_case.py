"""
CLI tool for adding labeled test cases to the eval ground truth.

Usage:
    python add_test_case.py --url <bluesky_url> --expected "<expected output>"

The expected output should be the bullet point explanation you would
manually write for this post, with each bullet on a new line.

Example:
    python add_test_case.py \\
        --url "https://bsky.app/profile/user.bsky.social/post/abc123" \\
        --expected "• Bullet point one
• Bullet point two
• Bullet point three"
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth" / "test_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a labeled test case to the eval ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        required=True,
        type=str,
        help="Full Bluesky post URL",
    )
    parser.add_argument(
        "--expected",
        required=True,
        type=str,
        help='Expected bullet point output (use \\n between bullets or multi-line string)',
    )
    return parser.parse_args()


def load_test_cases() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        return []
    with open(GROUND_TRUTH_PATH, "r") as f:
        return json.load(f)


def save_test_cases(cases: list[dict]) -> None:
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def generate_id(cases: list[dict]) -> str:
    """Generate a sequential test case ID like 'tc_001'."""
    next_num = len(cases) + 1
    return f"tc_{next_num:03d}"


def main() -> None:
    args = parse_args()

    cases = load_test_cases()

    # Check for duplicate URLs
    existing_urls = {c["url"] for c in cases}
    if args.url in existing_urls:
        print(f"Warning: A test case for this URL already exists.")
        print("Adding a duplicate entry. Remove the old one from test_cases.json if intended.")

    new_case = {
        "id": generate_id(cases),
        "url": args.url,
        "expected_output": args.expected,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }

    cases.append(new_case)
    save_test_cases(cases)

    print(f"Added test case {new_case['id']} for URL: {args.url}")
    print(f"Total test cases: {len(cases)}")


if __name__ == "__main__":
    main()
