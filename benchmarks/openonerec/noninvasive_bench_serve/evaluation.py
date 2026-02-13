"""
OpenOpenRec Evaluation Utilities

Metric computation functions for recommendation tasks.
Logic is identical to:
  - OpenOneRec/benchmarks/benchmark/tasks/v1_0/recommendation/utils.py
  - vllm_genrec/vllm/benchmarks/serve.py (evaluate_openopenrec section)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# SID extraction helpers
# ---------------------------------------------------------------------------


def extract_ids_from_answer(answer: str) -> set[str]:
    """Extract all SIDs from answer field.

    >>> extract_ids_from_answer("<|sid_begin|>123<|sid_end|><|sid_begin|>456<|sid_end|>")
    {'123', '456'}
    """
    correct_answers: set[str] = set()
    for part in answer.split('<|sid_begin|>'):
        if '<|sid_end|>' in part:
            sid = part.split('<|sid_end|>')[0].strip()
            if sid:
                correct_answers.add(sid)
    return correct_answers


def extract_first_id_from_answer(answer: str) -> str:
    """Extract the first SID from answer field."""
    for part in answer.split('<|sid_begin|>'):
        if '<|sid_end|>' in part:
            sid = part.split('<|sid_end|>')[0].strip()
            if sid:
                return sid
    return ""


def extract_id_from_generation(generation: str) -> str:
    """Extract SID from a single model generation string."""
    generation = generation.strip()

    # If generation contains </think>, only process content after it
    if '</think>' in generation:
        generation = generation.split('</think>')[-1].strip()

    # Try to extract from <|sid_begin|>...<|sid_end|> pattern
    if '<|sid_begin|>' in generation:
        for part in generation.split('<|sid_begin|>'):
            if '<|sid_end|>' in part:
                sid = part.split('<|sid_end|>')[0].strip()
                if sid:
                    return sid
            elif part.strip():  # No end marker, take the content after begin marker
                return part.strip()

    # Otherwise, return the stripped generation
    return generation


# ---------------------------------------------------------------------------
# Per-sample metric functions
# ---------------------------------------------------------------------------


def compute_pass_at_k(
    predicted_sids: list[str],
    ground_truth_sids: set[str],
    k: int,
) -> bool:
    """Pass@k: True if any of the first *k* predicted SIDs is in ground truth."""
    if not predicted_sids or not ground_truth_sids:
        return False
    top_k_sids = predicted_sids[:k]
    for sid in top_k_sids:
        if sid in ground_truth_sids:
            return True
    return False


def compute_position1_pass_at_k(
    predicted_sids: list[str],
    first_ground_truth_sid: str,
    k: int,
) -> bool:
    """Position1_Pass@k: True if any of the first *k* predicted SIDs matches
    the *first* ground-truth SID."""
    if not predicted_sids or not first_ground_truth_sid:
        return False
    top_k_sids = predicted_sids[:k]
    for sid in top_k_sids:
        if sid == first_ground_truth_sid:
            return True
    return False


def compute_recall_at_k(
    predicted_sids: list[str],
    ground_truth_sids: set[str],
    k: int,
) -> float:
    """Recall@k: fraction of ground-truth SIDs found in the first *k* predictions."""
    if not predicted_sids or not ground_truth_sids:
        return 0.0
    top_k_sids = predicted_sids[:k]
    predicted_sids_set = set(sid for sid in top_k_sids if sid)
    hit_count = len(predicted_sids_set & ground_truth_sids)
    recall = hit_count / len(ground_truth_sids)
    return recall


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate(
    samples: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    """Compute pass@k, position1_pass@k, recall@k over a list of samples.

    Each sample dict must have:
        - ``"groundtruth"`` (str): ground-truth answer with SID tags
        - ``"beams"`` (list[str]): beam search output texts
        - ``"sample_id"`` (str): unique identifier

    Uses exactly the same metric logic as
    ``OpenOneRec/benchmarks/benchmark/tasks/v1_0/recommendation/evaluator.py``.

    Returns a dict with overall metrics **and** a ``per_sample`` sub-dict.
    """
    total = len(samples)

    pass_counts = {k: 0 for k in k_values}
    pos1_pass_counts = {k: 0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    evaluated = 0
    skipped_no_gt = 0
    skipped_no_gen = 0
    per_sample: dict[str, dict[str, Any]] = {}

    for sample in samples:
        groundtruth = sample.get("groundtruth", "")
        sample_id = sample.get("sample_id", "")
        beams: list[str] = sample.get("beams", [])

        ground_truth_ids = extract_ids_from_answer(groundtruth)
        first_gt_id = extract_first_id_from_answer(groundtruth)

        if not ground_truth_ids:
            skipped_no_gt += 1
            continue

        if not beams:
            skipped_no_gen += 1
            continue

        # Extract predicted SIDs from beam outputs
        predicted_sids = [extract_id_from_generation(b) for b in beams]

        sample_metrics: dict[str, Any] = {}
        for k in k_values:
            p = compute_pass_at_k(predicted_sids, ground_truth_ids, k)
            p1 = compute_position1_pass_at_k(predicted_sids, first_gt_id, k)
            r = compute_recall_at_k(predicted_sids, ground_truth_ids, k)
            sample_metrics[f"pass@{k}"] = p
            sample_metrics[f"position1_pass@{k}"] = p1
            sample_metrics[f"recall@{k}"] = r
            if p:
                pass_counts[k] += 1
            if p1:
                pos1_pass_counts[k] += 1
            recall_sums[k] += r

        per_sample[sample_id] = sample_metrics
        evaluated += 1

    # Aggregate
    metrics: dict[str, Any] = {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "skipped_no_groundtruth": skipped_no_gt,
        "skipped_no_generation": skipped_no_gen,
    }
    for k in k_values:
        denom = evaluated if evaluated > 0 else 1
        metrics[f"pass@{k}"] = pass_counts[k] / denom
        metrics[f"position1_pass@{k}"] = pos1_pass_counts[k] / denom
        metrics[f"recall@{k}"] = recall_sums[k] / denom

    metrics["per_sample"] = per_sample
    return metrics
