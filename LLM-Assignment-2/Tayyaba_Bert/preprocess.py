import re
import json

def clean_text_safe(text):
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text

def get_spans(pred_seq, id2tag):
    spans, start, label = [], None, None
    for i, tag_id in enumerate(pred_seq):
        tag = id2tag[tag_id]
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-") and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i, label))
                start = None
    if start is not None:
        spans.append((start, len(pred_seq), label))
    return spans
