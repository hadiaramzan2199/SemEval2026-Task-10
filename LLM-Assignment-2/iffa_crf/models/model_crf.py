# models/model_crf.py

from typing import List, Dict
import os

import joblib
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn.metrics import classification_report

import nltk
from textblob import TextBlob

# Make sure NLTK POS tagger is available
nltk.download("averaged_perceptron_tagger", quiet=True)


def word_shape(token: str) -> str:
    """
    Rough word shape feature.
    Example: 'USA123!' -> 'XXXddd.'
    """
    shape = []
    for ch in token:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(".")
    return "".join(shape)


class CRFBaseline:
    """
    Baseline CRF model for BIO-tag marker extraction.
    Uses:
      - lexical features (token, prefix, suffix)
      - word shape
      - POS tags
      - sentence-level sentiment bin
    """

    def __init__(
        self,
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ):
        self.model = CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    # ---------- Feature extraction helpers ----------

    def _sent2features(self, sent: Dict) -> List[Dict]:
        """
        sent: dict with keys:
          - "tokens": List[str]
          - "labels": List[str]
          - "text":   str
        """
        tokens = sent["tokens"]
        text = sent["text"]

        # POS tags (list of (token, pos))
        pos_tags = [pos for (_, pos) in nltk.pos_tag(tokens)]

        # sentiment polarity (sentence-level)
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            sent_bin = "pos"
        elif polarity < -0.1:
            sent_bin = "neg"
        else:
            sent_bin = "neu"

        features_seq = []
        for i, token in enumerate(tokens):
            feats = {
                "bias": 1.0,
                "token.lower": token.lower(),
                "token.isupper": token.isupper(),
                "token.istitle": token.istitle(),
                "token.isdigit": token.isdigit(),
                "prefix1": token[:1],
                "prefix2": token[:2],
                "suffix1": token[-1:],
                "suffix2": token[-2:],
                "shape": word_shape(token),
                "pos": pos_tags[i],
                "sentiment_bin": sent_bin,
            }

            if i == 0:
                feats["BOS"] = True
            else:
                prev = tokens[i - 1]
                feats.update(
                    {
                        "-1:token.lower": prev.lower(),
                        "-1:token.isupper": prev.isupper(),
                        "-1:token.istitle": prev.istitle(),
                        "-1:pos": pos_tags[i - 1],
                    }
                )

            if i == len(tokens) - 1:
                feats["EOS"] = True
            else:
                nxt = tokens[i + 1]
                feats.update(
                    {
                        "+1:token.lower": nxt.lower(),
                        "+1:token.isupper": nxt.isupper(),
                        "+1:token.istitle": nxt.istitle(),
                        "+1:pos": pos_tags[i + 1],
                    }
                )

            features_seq.append(feats)

        return features_seq

    def _sent2labels(self, sent: Dict) -> List[str]:
        return sent["labels"]

    def _prepare_Xy(self, corpus: List[Dict]):
        X = [self._sent2features(s) for s in corpus]
        y = [self._sent2labels(s) for s in corpus]
        return X, y

    # ---------- Public API ----------

    def fit(self, train_corpus: List[Dict]):
        X_train, y_train = self._prepare_Xy(train_corpus)
        self.model.fit(X_train, y_train)

    def predict(self, corpus: List[Dict]) -> List[List[str]]:
        X, _ = self._prepare_Xy(corpus)
        return self.model.predict(X)

    def evaluate(self, corpus: List[Dict]) -> Dict:
        """
        Returns:
          {
            "f1_weighted": float,
            "f1_macro": float,
            "per_label_f1": Dict[str, float],
            "report_str": str
          }
        """
        X_val, y_true = self._prepare_Xy(corpus)
        y_pred = self.model.predict(X_val)

        # flatten for classification_report
        y_true_flat = [lab for seq in y_true for lab in seq]
        y_pred_flat = [lab for seq in y_pred for lab in seq]

        report_dict = classification_report(
            y_true_flat,
            y_pred_flat,
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            y_true_flat,
            y_pred_flat,
            zero_division=0
        )

        f1_weighted = report_dict["weighted avg"]["f1-score"]
        f1_macro = report_dict["macro avg"]["f1-score"]

        # per-label F1 (excluding averages)
        per_label_f1 = {
            label: v["f1-score"]
            for label, v in report_dict.items()
            if label not in {"accuracy", "macro avg", "weighted avg"}
        }

        return {
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "per_label_f1": per_label_f1,
            "report_str": report_str,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "CRFBaseline":
        model = joblib.load(path)
        obj = cls()
        obj.model = model
        return obj
