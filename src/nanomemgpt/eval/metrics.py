from __future__ import annotations

import re

from rouge_score import rouge_scorer


def rouge_l_recall(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].recall


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains_reference(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    return bool(normalized_reference and normalized_reference in normalized_prediction)
