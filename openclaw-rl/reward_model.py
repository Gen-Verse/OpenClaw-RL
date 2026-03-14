import logging
import os
import threading
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from feedback_store import FeedbackRecord, FeedbackStore

logger = logging.getLogger(__name__)

class RewardModel(nn.Module):
    """A lightweight reward model based on a transformer backbone."""
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.config = self.backbone.config
        # Simple linear head on top of the last hidden state of the first token (or mean pool)
        self.value_head = nn.Linear(self.config.hidden_size, 1).to(device)
        self.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use simple mean pooling across sequence length
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        value = self.value_head(pooled_output)
        return value.squeeze(-1)

    @torch.no_grad()
    def get_reward(self, prompt: str, response: str, tokenizer: PreTrainedTokenizer) -> float:
        self.eval()
        text = f"Prompt: {prompt}\n\nResponse: {response}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        reward = self.forward(**inputs).item()
        # Scale to [-1, 1] approximately if needed, or keep raw
        return float(reward)

class FeedbackDataset(Dataset):
    def __init__(self, records: List[FeedbackRecord], tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = f"Prompt: {rec.prompt_text}\n\nResponse: {rec.response_text}"
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(float(rec.label), dtype=torch.float)
        }

class RewardModelManager:
    """Manages the lifecycle, inference, and background training of the Reward Model."""
    def __init__(
        self,
        model_path: str,
        feedback_store: FeedbackStore,
        device: str = "cuda",
        lr: float = 1e-5,
        batch_size: int = 8,
        train_interval: int = 300, # seconds
    ):
        self.feedback_store = feedback_store
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = RewardModel(model_path, device=device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.running = False
        self._train_thread: Optional[threading.Thread] = None

    def start_background_training(self):
        if self.running:
            return
        self.running = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()
        logger.info("[RewardModel] Background training started.")

    def stop_background_training(self):
        self.running = False
        if self._train_thread:
            self._train_thread.join(timeout=5)
        logger.info("[RewardModel] Background training stopped.")

    def _train_loop(self):
        while self.running:
            try:
                # Only train if we have both positive and negative samples
                if self.feedback_store.num_negatives > 0 and self.feedback_store.num_positives > 0:
                    self._do_train_step()
            except Exception as e:
                logger.error(f"[RewardModel] Training step failed: {e}")
            
            # Wait for next interval
            for _ in range(self.train_interval):
                if not self.running: break
                time.sleep(1)

    def _do_train_step(self):
        # Sample records from store
        records = self.feedback_store.sample_batch(batch_size=self.batch_size * 4, balance=True)
        if len(records) < 4:
            return

        dataset = FeedbackDataset(records, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["label"].to(self.model.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        logger.info(f"[RewardModel] Train step completed. Avg Loss: {total_loss/len(dataloader):.4f}")

    def score(self, prompt: str, response: str) -> float:
        return self.model.get_reward(prompt, response, self.tokenizer)
