import wandb
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils import inf_loop


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        data_loaders,
        device,
        n_epochs: int,
        len_epoch: int = None,
        log_step: int = 50,
        grad_norm_clip = None,
        wandb_project: str = 'tiny-lm',
        wandb_run: str = 'one_batch_test',
        **kwargs
    ):
        assert 'train' in data_loaders and 'valid' in data_loaders
        
        self.device = device
        self.model = model.to(device)
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        
        self.train_loader = data_loaders['train']
        self.valid_loader = data_loaders['valid']
        
        self.grad_norm_clip = grad_norm_clip
        self.n_epochs = n_epochs
        
        # ----------LOGGING----------
        self.setup_logger(wandb_project, wandb_run)
        self.log_step = log_step
        self.train_losses = []
        self.valid_losses = []
        self.grad_norms = []
        if len_epoch is None:
            self.len_epoch = len(self.train_loader)
        else:
            self.train_loader = inf_loop(self.train_loader)
            self.len_epoch = len_epoch 
        
    def setup_logger(self, project_name, run_name):
        wandb.login()
        wandb.init(
            project=project_name,
            name=run_name
        )
        self.logger = wandb
    
    def train_epoch(self, epoch):
        self.model.train()
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f'Training epoch {epoch + 1}/{self.n_epochs}', 
                 total=self.len_epoch)
        ):
            if batch_idx == self.len_epoch:
                break

            input_ids = batch['input_ids'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            logits = self.model(input_ids, padding_mask)
            loss = self.criterion(logits, input_ids)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_norm_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            self.train_losses += [loss.detach().cpu().numpy()]
            self.grad_norms += [self.get_grad_norm()]
            
            if batch_idx % self.log_step == 0:
                step = epoch * self.len_epoch
                mean_loss = sum(self.train_losses) / len(self.train_losses)
                mean_grad_norm = sum(self.grad_norms) / len(self.grad_norms)
                
                self.logger.log({"train_loss": mean_loss}, step=step)
                self.logger.log({"learning_rate": self.lr_scheduler.get_last_lr()[0]}, step=step)
                self.logger.log({"grad_norm": mean_grad_norm}, step=step)
                
                self.train_losses.clear()
                self.grad_norms.clear()
    
    def eval_epoch(self, epoch):
        self.model.eval()
        
        for batch_idx, batch in enumerate(
            tqdm(self.valid_loader, desc=f'Evaluating epoch {epoch + 1}/{self.n_epochs}')
        ):
            input_ids = batch['input_ids'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            logits = self.model(input_ids, padding_mask)
            
            loss = self.criterion(logits, input_ids)
            
            self.valid_losses += [loss.detach().cpu().numpy()]
            
        step = epoch * self.len_epoch
        mean_loss = sum(self.valid_losses) / len(self.valid_losses)
        # print(f'Losses: {self.valid_losses}')
        self.logger.log({"valid_loss": mean_loss}, step=step)
        self.valid_losses.clear()

    def train(self):
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        parameters = [p for p in parameters if p.grad is not None]
        params_norms = [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        assert len(params_norms) > 0, parameters
        
        total_norm = torch.norm(torch.stack(params_norms), norm_type)
        
        return total_norm.detach().cpu().numpy()
