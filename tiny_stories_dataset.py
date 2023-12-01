import os
import time
import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

from utils import preprocess_txt


class TinyStories(Dataset):
    def __init__(
        self,
        root_path: str = 'data',
        split: str = 'train',
        max_len: int = 512,
        limit: int = None,
        vocab_size: int = None,
        **kwargs
    ):
        super().__init__()
        assert split in ('train', 'test')
        self.root = root_path
        self.max_len = max_len - 2  # w/o bos/eos tokens

        path_to_data = preprocess_txt(self.root, split)
        with open(path_to_data, 'r') as f:
            self.data = f.readlines()

        if limit is not None and limit < len(self.data):
            self.data = self.data[:limit]

        self.tokenizer = self._get_tokenizer(vocab_size)
    
    def _get_tokenizer(self, vocab_size):
        if not os.path.exists('data/sp.model'):
            self._train_tokenizer(vocab_size)
            
        return SentencePieceProcessor(model_file='data/sp.model')
    
    def _train_tokenizer(self, vocab_size: int = None):
        if vocab_size is None:
            vocab_size = 4000
        input_file = preprocess_txt(self.root, 'train')
        print('Training sentencepiece tokenizer...')
        start = time.perf_counter()
        SentencePieceTrainer.train(
            input=input_file, vocab_size=4000,
            model_type='bpe', model_prefix=f'{self.root}/sp',
            normalization_rule_name='nmt_nfkc_cf',
            pad_id=3
        )
        end = time.perf_counter()
        print(f'Trained sentencepiece tokenizer in {end - start} seconds.')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        indices = self.tokenizer.encode(self.data[idx])
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]

        input_ids = [self.tokenizer.bos_id()] + indices + [self.tokenizer.eos_id()]
        input_ids = torch.tensor(input_ids, dtype=torch.float)

        return {
            "input_ids": input_ids
        }
