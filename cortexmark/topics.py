"""Topic classification for Markdown documents.

Uses keyword-frequency heuristics to assign topic labels:
  - Machine Learning, Reinforcement Learning, NLP, Computer Vision,
    Optimization, Statistics, Mathematics, Physics, Economics, Generic

Provides:
  - Per-document topic scoring and top-N labels
  - Per-tree aggregated topic distribution
  - JSON report output
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Topic keyword dictionaries ───────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "reinforcement_learning": [
        "reinforcement learning",
        "markov decision process",
        "mdp",
        "policy gradient",
        "q-learning",
        "bellman equation",
        "reward",
        "temporal difference",
        "td learning",
        "actor-critic",
        "value function",
        "action-value",
        "epsilon-greedy",
        "bandit",
        "sarsa",
        "monte carlo",
        "exploration",
        "exploitation",
        "discount factor",
        "trajectory",
        "episode",
    ],
    "machine_learning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "classification",
        "regression",
        "neural network",
        "deep learning",
        "gradient descent",
        "backpropagation",
        "overfitting",
        "cross-validation",
        "training set",
        "test set",
        "feature",
        "loss function",
        "batch normalization",
        "dropout",
    ],
    "nlp": [
        "natural language processing",
        "nlp",
        "language model",
        "transformer",
        "attention mechanism",
        "tokenization",
        "word embedding",
        "bert",
        "gpt",
        "seq2seq",
        "sentiment analysis",
        "named entity",
        "part-of-speech",
        "text classification",
    ],
    "computer_vision": [
        "computer vision",
        "image classification",
        "object detection",
        "convolutional neural network",
        "cnn",
        "image segmentation",
        "feature extraction",
        "pixel",
        "bounding box",
        "resnet",
        "generative adversarial",
        "gan",
    ],
    "optimization": [
        "optimization",
        "convex optimization",
        "linear programming",
        "gradient descent",
        "stochastic gradient",
        "lagrange multiplier",
        "constraint",
        "objective function",
        "convergence",
        "dynamic programming",
        "heuristic",
    ],
    "statistics": [
        "statistics",
        "probability",
        "distribution",
        "hypothesis test",
        "confidence interval",
        "regression analysis",
        "variance",
        "standard deviation",
        "bayesian",
        "maximum likelihood",
        "p-value",
        "statistical significance",
        "sampling",
    ],
    "mathematics": [
        "theorem",
        "proof",
        "lemma",
        "corollary",
        "definition",
        "linear algebra",
        "eigenvalue",
        "matrix",
        "vector space",
        "calculus",
        "integral",
        "differential equation",
        "topology",
        "manifold",
    ],
    "physics": [
        "physics",
        "quantum",
        "hamiltonian",
        "lagrangian",
        "wave function",
        "schrodinger",
        "thermodynamics",
        "entropy",
        "electrodynamics",
        "relativity",
    ],
    "economics": [
        "economics",
        "market",
        "supply and demand",
        "equilibrium",
        "utility",
        "marginal",
        "inflation",
        "gdp",
        "monetary policy",
        "fiscal policy",
        "game theory",
    ],
}


@dataclass
class TopicScore:
    """Score for a single topic."""

    topic: str
    score: float
    keyword_hits: int


@dataclass
class DocumentTopics:
    """Topic classification result for a single document."""

    source_file: str
    word_count: int
    scores: list[TopicScore] = field(default_factory=list)
    primary_topic: str = "generic"
    confidence: float = 0.0


# ── Scoring engine ───────────────────────────────────────────────────────────


def classify_text(text: str, source_file: str = "") -> DocumentTopics:
    """Classify a document's topics based on keyword frequency."""
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    if word_count == 0:
        return DocumentTopics(
            source_file=source_file,
            word_count=0,
            primary_topic="generic",
        )

    topic_scores: list[TopicScore] = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            # Count occurrences of multi-word or single-word keywords
            hits += len(re.findall(rf"\b{re.escape(kw)}\b", text_lower))

        if hits > 0:
            # Normalize by document length (per 1000 words)
            score = round((hits / word_count) * 1000, 2)
            topic_scores.append(TopicScore(topic=topic, score=score, keyword_hits=hits))

    topic_scores.sort(key=lambda t: t.score, reverse=True)

    top_score = topic_scores[0] if topic_scores else None
    if top_score is None or top_score.keyword_hits < 2:
        primary = "generic"
        confidence = 0.0
    else:
        primary = top_score.topic
        total_hits = sum(topic.keyword_hits for topic in topic_scores)
        evidence_strength = min(top_score.keyword_hits / 5.0, 1.0)
        confidence = round(evidence_strength * (top_score.keyword_hits / total_hits), 3)

    return DocumentTopics(
        source_file=source_file,
        word_count=word_count,
        scores=topic_scores,
        primary_topic=primary,
        confidence=confidence,
    )


def get_top_topics(doc: DocumentTopics, n: int = 3) -> list[str]:
    """Return the top-N topic labels for a document."""
    return [s.topic for s in doc.scores[:n]]


# ── File / tree operations ───────────────────────────────────────────────────


def classify_file(file_path: Path) -> DocumentTopics:
    """Classify a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return classify_text(text, source_file=str(file_path))


def classify_tree(input_root: Path) -> list[DocumentTopics]:
    """Classify all Markdown files in a directory tree."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [classify_file(f) for f in md_files]


def build_topic_distribution(results: list[DocumentTopics]) -> dict[str, int]:
    """Count how many documents have each primary topic."""
    counter: Counter[str] = Counter()
    for r in results:
        counter[r.primary_topic] += 1
    return dict(counter.most_common())


def write_topic_report(results: list[DocumentTopics], output_path: Path) -> Path:
    """Write topic classification report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dist = build_topic_distribution(results)
    data: dict[str, Any] = {
        "summary": {
            "files_scanned": len(results),
            "topic_distribution": dist,
        },
        "files": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify topics of Markdown documents.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for topic report (JSON)")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("topics", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "topics.json")).resolve()

    try:
        results = classify_tree(input_path) if input_path.is_dir() else [classify_file(input_path)]

        written = write_topic_report(results, output_path)
        dist = build_topic_distribution(results)
        log.info(
            "classified %d file(s): %s -> %s",
            len(results),
            ", ".join(f"{t}={c}" for t, c in dist.items()),
            written,
        )

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
