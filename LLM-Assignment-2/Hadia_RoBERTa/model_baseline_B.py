"""
RoBERTa Baseline for Sequence Labeling (Subtask 1)
Student: Hadia
Task: Subtask 1 - Extract psycholinguistic markers (BIO tags)
Model: XLM-RoBERTa-base
"""

from model_roberta_ner import RoBERTaSequenceTagger

class RoBERTaBaseline:
    def __init__(self, model_name="xlm-roberta-base", device=None):
        self.model = RoBERTaSequenceTagger(model_name=model_name, device=device)
    
    def fit(self, train_corpus, val_corpus=None):
        return self.model.train(train_corpus, val_corpus)
    
    def predict(self, corpus):
        return self.model.predict(corpus)
    
    def evaluate(self, corpus):
        return self.model.evaluate(corpus)
    
    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path, device=None):
        model = cls(device=device)
        model.model = RoBERTaSequenceTagger.load(path, device)
        return model

if __name__ == "__main__":
    print("RoBERTa Baseline Model for Sequence Labeling")
    print("Task: Subtask 1 - Extract psycholinguistic markers")
    print("Labels: 11 BIO tags (5 marker types × B/I + O)")
