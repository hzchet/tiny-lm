from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader

from tiny_stories_dataset import TinyStories
from model import LanguageModel
from utils import LMCrossEntropyLoss, collate_fn
from trainer import Trainer


@dataclass
class Config:
    root_path: str = 'data'
    max_len: int = 64
    limit: int = None
    vocab_size: int = 4000
    
    batch_size: int = 32
    
    embedding_dim: int = 32
    num_embeddings: int = 4000
    num_encoder_layers: int = 1
    n_heads: int = 4
    dim_feedforward: int = 64
    
    lr: float = 3e-4
    
    grad_norm_clip = 1.0
    n_epochs: int = 100
    len_epoch: int = 100
    log_step: int = 50
    wandb_project: str = 'tiny-lm'
    wandb_run: str = 'one_batch_test'
    

def one_batch_test():
    cfg = Config(limit=10, batch_size=10)
    
    dataset = TinyStories(cfg.root_path, 'train', cfg.max_len, cfg.limit)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, 
                              collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, 
                              collate_fn=collate_fn, shuffle=False)
    data_loaders = {"train": train_loader, "valid": valid_loader}
    
    model = LanguageModel(**asdict(cfg))
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total number of parameters: {total_params}")
    
    optimizer = torch.optim.Adam(trainable_params, cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.n_epochs, steps_per_epoch=cfg.len_epoch,
        anneal_strategy='cos', pct_start=0.2
    )
    criterion = LMCrossEntropyLoss()
    
    trainer = Trainer(
        model, 
        optimizer, 
        lr_scheduler, 
        criterion, 
        data_loaders,
        device='cuda:0', 
        **asdict(cfg)
    )
    trainer.train()


if __name__ == '__main__':
    one_batch_test()
