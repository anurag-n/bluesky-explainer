"""
Bluesky Post Explainer — CLI entry point.

Usage:
    python main.py <bluesky_post_url> [--verbose]

Example:
    python main.py "https://bsky.app/profile/user.bsky.social/post/abc123"
    python main.py "https://bsky.app/profile/user.bsky.social/post/abc123" --verbose
"""

import argparse
import sys
from pathlib import Path

# Allow running directly from the agent/ directory (python main.py ...)
# without needing to install the package.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.pipeline import ExplainerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a Bluesky post with AI-powered context extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py 'https://bsky.app/profile/user.bsky.social/post/abc123'\n"
            "  python main.py 'https://bsky.app/profile/user.bsky.social/post/abc123' -v\n"
        ),
    )
    parser.add_argument(
        "post_url",
        type=str,
        help="Full Bluesky post URL (e.g. https://bsky.app/profile/handle/post/rkey)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print intermediate pipeline steps to stdout",
    )
    return parser.parse_args()


def main() -> None:
    # Load .env from the project root (one level up from agent/)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    args = parse_args()

    try:
        pipeline = ExplainerPipeline(verbose=args.verbose)
        result = pipeline.run(post_url=args.post_url)
        print(result)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
