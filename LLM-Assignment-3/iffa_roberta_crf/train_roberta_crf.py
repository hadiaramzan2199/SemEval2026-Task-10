import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_crf import RoBERTaCRF

MODEL_NAME = "roberta-base"

LABELS = [
    "O",
    "B-ACTOR", "I-ACTOR",
    "B-ACTION", "I-ACTION",
    "B-EFFECT", "I-EFFECT",
    "B-EVIDENCE", "I-EVIDENCE",
    "B-VICTIM", "I-VICTIM"
]

label2id = {l: i for i, l in enumerate(LABELS)}

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    add_prefix_space=True
)


def encode(corpus):
    input_ids, masks, labels = [], [], []

    for s in corpus:
        enc = tokenizer(
            s["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        word_ids = enc.word_ids()
        label_ids = []
        prev = None

        for w in word_ids:
            if w is None or w == prev:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[s["labels"][w]])
            prev = w

        input_ids.append(enc["input_ids"][0])
        masks.append(enc["attention_mask"][0])
        labels.append(torch.tensor(label_ids))

    return torch.stack(input_ids), torch.stack(masks), torch.stack(labels)

def main():
    df, _ = load_psycomark(
    "/content/SemEval2026-Task-10/LLM-Assignment-2/data/train_rehydrated.jsonl"
)

    corpus = build_bio_corpus(df)
    train_c, val_c = train_val_split(corpus)

    X_train, M_train, Y_train = encode(train_c)

    device = torch.device("cuda")
    model = RoBERTaCRF(MODEL_NAME, len(LABELS)).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        out = model(
            X_train.to(device),
            M_train.to(device),
            Y_train.to(device)
        )
        out["loss"].backward()
        optimizer.step()
        print(f"Epoch {epoch+1} loss:", out["loss"].item())

    torch.save(model.state_dict(), "results/roberta_crf.pth")

if __name__ == "__main__":
    main()
