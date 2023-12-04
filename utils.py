import os
import time
from typing import List
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def preprocess_txt(root: str, split: str):
    preprocessed_txt = f'{root}/{split}_preprocessed.txt'
    if os.path.exists(preprocessed_txt):
        print(f'{split} text is already preprocessed.')
        return preprocessed_txt
    
    print(f'Preprocessing {split} text...')
    start = time.perf_counter()
    with open(f'{root}/{split}.txt', 'r', encoding='utf-8') as input_file:
        current_story = []
        for line in tqdm(input_file):
            if '<|endoftext|>' in line:
                current_story = ' '.join(current_story)
                if current_story is not None:
                    with open(preprocessed_txt, 'a') as f:
                        f.write(current_story + '\n')
                current_story = []
            else:
                if line.strip():
                    current_story.append(line.strip())
    
    end = time.perf_counter()
    print(f'Preprocessed {split} text in {end - start} seconds.')
    
    return preprocessed_txt


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    
    input_ids = [item['input_ids'] for item in dataset_items]
    padding_mask = [torch.ones(len(item['input_ids'])) for item in dataset_items]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(torch.long)
    padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'padding_mask': padding_mask
    }


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


class LMCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=3)
    
    def forward(self, logits, tokens):
        """
        logits: torch.Tensor (B, Len, |V|)
        tokens: torch.Tensor (B, Len)
        """
        return self.ce(logits[:, :-1].transpose(1, 2), tokens[:, 1:])


def generate(model, tokenizer, batch_size: int, prefix=None, max_len: int = 100, device='cpu'):
    if prefix is None:
        prefix = torch.tensor([[tokenizer.bos_id()] for _ in range(batch_size)]).to(device)
    else:
        prefix = torch.tensor([tokenizer.encode(prefix) for _ in range(batch_size)]).to(device)
    
    output = prefix.clone()
    
    for _ in range(max_len):
        logits = model(output, None)
        
        next_token_probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(next_token_probs, 1)

        output = torch.cat((output, next_token), dim=1)
        
    return tokenizer.decode(output.tolist())
