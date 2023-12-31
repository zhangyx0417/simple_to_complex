import json

import torch
from torch.utils.data import Dataset


def my_collate(batch):
    event_idx = [ex["event_idx"] for ex in batch]
    doc_key = [ex["doc_key"] for ex in batch]
    event_type = [ex["event_type"] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex["input_token_ids"]) for ex in batch])
    input_attn_mask = torch.stack([torch.BoolTensor(ex["input_attn_mask"]) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex["tgt_token_ids"]) for ex in batch])
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex["tgt_attn_mask"]) for ex in batch])

    input_template = [ex["input_template"] for ex in batch]
    context_tokens = [ex["context_tokens"] for ex in batch]
    context_words = [ex["context_words"] for ex in batch]

    return {
        "event_idx": event_idx,
        "input_token_ids": input_token_ids,
        "event_type": event_type,
        "input_attn_mask": input_attn_mask,
        "tgt_token_ids": tgt_token_ids,
        "tgt_attn_mask": tgt_attn_mask,
        "doc_key": doc_key,
        "input_template": input_template,
        "context_tokens": context_tokens,
        "context_words": context_words,
    }


class IEDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.examples = []
        with open(input_file, "r") as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
