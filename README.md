# Bluesky Post Explainer

An AI agent that explains Bluesky social media posts by extracting their full context (text, images, linked articles, quoted posts) and returning 3–5 bullet points of relevant background.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Setup](#setup)
- [Running the Agent](#running-the-agent)
- [Running the Eval Harness](#running-the-eval-harness)
- [Adding Test Cases](#adding-test-cases)

---

## Project Overview

```
bluesky-explainer/
├── agent/               # The explainer agent
│   ├── src/             # Core classes
│   ├── prompts/         # YAML prompt configs for all LLM calls
│   └── main.py          # CLI entry point
├── eval/                # Evaluation harness
│   ├── src/             # Similarity scorer + LLM judge
│   ├── prompts/         # Judge prompt config
│   ├── ground_truth/    # Labeled test cases (JSON)
│   ├── add_test_case.py # Tool to add ground truth
│   └── evaluate.py      # Evaluation script
└── .env.template        # Environment variable template
```

---

## Architecture & Design Decisions

### Agent Pipeline

The agent runs three sequential steps:

**Step 1 — Data Extraction**

Fetches the post via the Bluesky AT Protocol API (`atproto` library) and extracts all content:

- **Post text**: Summarized and translated to English if needed.
- **Images**: Described using Claude's vision capability via a multimodal LLM call.
- **External URLs** (link cards): Scraped with BeautifulSoup, then summarized by the LLM.
- **Quoted posts**: Fetched via the Bluesky API and summarized.

All summaries are combined into a single text block for the next steps.

**Step 2 — Guardrail Check**

The combined content is passed to an LLM that checks for profanity, hate speech, harassment, and explicit content. The LLM returns a structured JSON response (`{"result": "PASS"}` or `{"result": "FAIL", "reason": "..."}`). If the content fails, the pipeline stops and returns:

> Post did not pass profanity filter

**Step 3 — Explanation**

The combined content is passed to a final LLM call that returns exactly 3–5 bullet points of contextual background explaining the post's topic to a reader unfamiliar with it. Each response must include at least one inline citation in `(Source: ...)` format, pointing to a news outlet, Wikipedia, an official organisation, or a URL extracted from the post itself.

### LLM Strategy

| Role | Model | Provider |
|------|-------|----------|
| Primary (text + vision) | `gpt-4o` | OpenAI |
| Fallback (context overflow) | `gpt-4o-mini` | OpenAI |

A single API key is used for both models. The fallback is triggered only on context-window errors (e.g., very long scraped articles). Both models support vision via the same `image_url` message format, so the same fallback handles text and image calls.

LangChain LCEL is used to build composable prompt chains:
```
ChatPromptTemplate | LLM | StrOutputParser
```

### Prompt Configuration

All LLM calls are driven by YAML files in `agent/prompts/`. Each file specifies:
- `system`: System prompt template
- `user`: User prompt template (uses `{input}` placeholder)
- `model_params`: `temperature`, `max_tokens`

This makes prompts easy to iterate on without changing any Python code.

### Eval Harness Design

The eval harness is **self-contained** — it imports the agent's pipeline but has its own LLM client and prompts. This allows the eval to run independently.

Two scoring mechanisms are combined:

1. **Embedding similarity** (`sentence-transformers`, `all-MiniLM-L6-v2`): Cosine similarity between the agent's output and the manually-written expected output. Measures semantic closeness.

2. **LLM-as-a-judge** (GPT-4o): Evaluates four binary metrics:
   - **Relevance** (0 or 1): Was the context relevant to the post topic?
   - **Formatting** (0 or 1): Did the output consist only of bullet points?
   - **Length** (0 or 1): Was the output 3–5 bullet points?
   - **Citation** (0 or 1): Does the output contain at least one inline citation?

---

## Setup

### Prerequisites

- Python 3.11+
- A [Bluesky account](https://bsky.app) with an [app password](https://bsky.app/settings/app-passwords)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anurag-n/bluesky-explainer.git
cd bluesky-explainer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. Install agent dependencies
pip install -r agent/requirements.txt

# 4. (Optional) Install eval dependencies
pip install -r eval/requirements.txt

# 5. Set up environment variables
cp .env.template .env
# Open .env and fill in your credentials
```

### Environment Variables

Edit `.env` with your credentials:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_FALLBACK_MODEL=gpt-4o-mini

BLUESKY_HANDLE=yourname.bsky.social
BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

---

## Running the Agent

```bash
# From the project root
python agent/main.py "https://bsky.app/profile/user.bsky.social/post/abc123"

# With verbose output (shows intermediate pipeline steps)
python agent/main.py "https://bsky.app/profile/user.bsky.social/post/abc123" --verbose
```

**Example output:**
```
• The post discusses the recent discovery of a new exoplanet in the habitable zone of a nearby star. (Source: NASA)
• The habitable zone (also called the "Goldilocks zone") is the range of orbital distances where liquid water can exist on a planet's surface.
• NASA's James Webb Space Telescope, launched in December 2021, is the primary instrument used for such atmospheric analyses.
• Finding planets in habitable zones is a key step in the search for extraterrestrial life, though habitability depends on many other factors.
```

---

## Running the Eval Harness

```bash
# Run evaluation on all labeled test cases
python eval/evaluate.py

# With verbose output
python eval/evaluate.py --verbose
```

**Example output:**
```
Loaded 3 test case(s).
Initializing pipeline and evaluators...

=================================================================
ID       Similarity  Relevance Formatting  Length  Citation  Average
=================================================================

[tc_001] Running agent on: https://bsky.app/...
tc_001        0.812        1.0        1.0     1.0       1.0    0.953

[tc_002] Running agent on: https://bsky.app/...
tc_002        0.743        1.0        0.0     1.0       0.0    0.686
-----------------------------------------------------------------
AVERAGE       0.778        1.0        0.5     1.0       0.5    0.820
=================================================================

Evaluated 2 test case(s).
  Embedding similarity (vs ground truth):  0.778
  Relevance (LLM judge, 0-1):              1.000
  Formatting (LLM judge, 0-1):             0.500
  Length (LLM judge, 0-1):                1.000
  Citation (LLM judge, 0-1):               0.500
  Overall average:                          0.820
```

---

## Adding Test Cases

```bash
python eval/add_test_case.py \
    --url "https://bsky.app/profile/user.bsky.social/post/abc123" \
    --expected "• Bullet point one
• Bullet point two
• Bullet point three"
```

Test cases are stored in `eval/ground_truth/test_cases.json`. You can also edit this file directly:

```json
[
  {
    "id": "tc_001",
    "url": "https://bsky.app/profile/user.bsky.social/post/abc123",
    "expected_output": "• Bullet point one\n• Bullet point two\n• Bullet point three",
    "added_at": "2026-02-25T00:00:00+00:00"
  }
]
```
