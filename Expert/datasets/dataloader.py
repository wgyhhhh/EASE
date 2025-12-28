import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np

label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "unknown": 2,
    0: 0,
    1: 1,
    2: 2
}


def word2input(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


class FakeNewsDataset(Dataset):
    def __init__(self, data_path, max_len, bert_path):
        super().__init__()
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Text encoding for content
        text = item.get('content', '')
        content_token_ids, content_masks = word2input([text], self.max_len, self.tokenizer)
        content_token_ids = content_token_ids.squeeze(0)
        content_masks = content_masks.squeeze(0)

        # Text encoding for expert features
        ftr_2_text = item.get('sentiment', '')
        ftr_2_token_ids, ftr_2_masks = word2input([ftr_2_text], self.max_len, self.tokenizer)
        ftr_2_token_ids = ftr_2_token_ids.squeeze(0)
        ftr_2_masks = ftr_2_masks.squeeze(0)

        ftr_3_text = item.get('reasoning', '')
        ftr_3_token_ids, ftr_3_masks = word2input([ftr_3_text], self.max_len, self.tokenizer)
        ftr_3_token_ids = ftr_3_token_ids.squeeze(0)
        ftr_3_masks = ftr_3_masks.squeeze(0)

        ftr_4_text = item.get('evidence', '')
        ftr_4_token_ids, ftr_4_masks = word2input([ftr_4_text], self.max_len, self.tokenizer)
        ftr_4_token_ids = ftr_4_token_ids.squeeze(0)
        ftr_4_masks = ftr_4_masks.squeeze(0)

        # Labels
        raw_label = item.get('label', 0)
        label = torch.tensor(label_dict.get(raw_label, raw_label), dtype=torch.float)

        # Expert features
        ftr_2_pred_raw = item.get('sentiment_pred', 0)
        ftr_2_pred = torch.tensor(label_dict_ftr_pred.get(ftr_2_pred_raw, ftr_2_pred_raw), dtype=torch.long)

        ftr_3_pred_raw = item.get('reasoning_pred', 0)
        ftr_3_pred = torch.tensor(label_dict_ftr_pred.get(ftr_3_pred_raw, ftr_3_pred_raw), dtype=torch.long)

        ftr_4_pred_raw = item.get('evidence_pred', 0)
        ftr_4_pred = torch.tensor(label_dict_ftr_pred.get(ftr_4_pred_raw, ftr_4_pred_raw), dtype=torch.long)

        # Accuracy features
        ftr_2_acc = torch.tensor(float(item.get('sentiment_acc', 0)), dtype=torch.float)
        ftr_3_acc = torch.tensor(float(item.get('reasoning_acc', 0)), dtype=torch.float)
        ftr_4_acc = torch.tensor(float(item.get('evidence_acc', 0)), dtype=torch.float)

        # Additional features
        evi_reliable = torch.tensor(float(item.get('evidence_reliable', 0)), dtype=torch.float)
        rea_reliable = torch.tensor(float(item.get('reasoning_reliable', 0)), dtype=torch.float)

        # ID for tracking
        item_id = item.get('id', idx)

        return {
            'content': content_token_ids,
            'content_masks': content_masks,
            'FTR_2': ftr_2_token_ids,
            'FTR_2_masks': ftr_2_masks,
            'FTR_3': ftr_3_token_ids,
            'FTR_3_masks': ftr_3_masks,
            'FTR_4': ftr_4_token_ids,
            'FTR_4_masks': ftr_4_masks,
            'label': label,
            'FTR_2_pred': ftr_2_pred,
            'FTR_2_acc': ftr_2_acc,
            'FTR_3_pred': ftr_3_pred,
            'FTR_3_acc': ftr_3_acc,
            'FTR_4_pred': ftr_4_pred,
            'FTR_4_acc': ftr_4_acc,
            'evi_reliable': evi_reliable,
            'rea_reliable': rea_reliable,
            'id': item_id
        }


def get_dataloader(data_path, max_len, batch_size, shuffle=False, bert_path=None, data_type='rationale', language=None):
    """Get data loader for fake news detection"""
    dataset = FakeNewsDataset(data_path, max_len, bert_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return dataloader


def collate_fn(batch):
    """Custom collate function for batching"""
    batch_dict = {}
    for key in batch[0].keys():
        if key == 'id':
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict