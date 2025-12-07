# preprocess.py

from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple whitespace-based tokenizer that also returns character offsets.
    Returns list of (token, start_char, end_char).
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        # skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        start = i
        while i < n and not text[i].isspace():
            i += 1
        end = i
        tokens.append((text[start:end], start, end))

    return tokens


def align_markers_to_tokens(
    text: str,
    markers: List[Dict]
) -> Tuple[List[str], List[str]]:
    """
    Convert a text + list of marker spans to (tokens, BIO-labels).

    markers: list of dicts with keys:
      - startIndex (int)
      - endIndex (int)
      - type (str) e.g. "Actor", "Action", "Effect", "Evidence", "Victim"
      - text (str) [optional]
    """
    tokens_offsets = tokenize_with_offsets(text)
    tokens = [t[0] for t in tokens_offsets]
    labels = ["O"] * len(tokens)

    for m in markers:
        m_type = m.get("type", "")
        if not m_type:
            continue
        tag = m_type.upper()

        span_start = m.get("startIndex", None)
        span_end = m.get("endIndex", None)
        if span_start is None or span_end is None:
            continue

        inside = False
        for i, (_, tok_start, tok_end) in enumerate(tokens_offsets):
            # overlap check between token [tok_start, tok_end) and span [span_start, span_end)
            if tok_end <= span_start or tok_start >= span_end:
                continue

            # if token already labeled, we keep first assignment (simple conflict resolution)
            if labels[i] != "O":
                continue

            if not inside:
                labels[i] = f"B-{tag}"
                inside = True
            else:
                labels[i] = f"I-{tag}"

    return tokens, labels


def build_bio_corpus(df) -> List[Dict]:
    """
    Build a BIO-tagged corpus from DataFrame.

    Returns:
      corpus: list of dict
        each dict has:
          - "tokens": List[str]
          - "labels": List[str]  (BIO tags)
          - "text":   str        (full original text)
    """
    corpus = []
    for _, row in df.iterrows():
        text = row["text"]
        markers = row.get("markers", [])
        tokens, labels = align_markers_to_tokens(text, markers)

        corpus.append(
            {
                "tokens": tokens,
                "labels": labels,
                "text": text,
            }
        )
    return corpus


def train_val_split(
    corpus: List[Dict],
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split the corpus into train and validation sets.
    """
    return train_test_split(corpus, test_size=test_size, random_state=random_state)
