import glob
import os
from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch.nn as nn
import lightning as L
from llm_components.text_summarization.text_summarization import TextSummarization, TextSummarizationDataModule
from tqdm import tqdm

class TLDR(L.LightningWork, ABC):
    """Finetune on a text summarization task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drive = L.app.storage.Drive("lit://artifacts")

    @abstractmethod
    def get_model(self) -> Tuple[nn.Module, Any]:
        """Return your large transformer language model here."""

    @abstractmethod
    def get_data_source(self) -> str:
        """Return a path to a file or a public URL that can be downloaded."""

    def get_trainer_settings(self):
        """Override this to change the Lightning Trainer default settings for finetuning."""
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            verbose=True,
            mode="min",
        )
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            # dirpath="drive",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        return dict(max_epochs=2, limit_train_batches=1, limit_val_batches=1, callbacks=[early_stopping, checkpoints], strategy="ddp_find_unused_parameters_false")

    def run(self):
        # for huggingface/transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        module, tokenizer = self.get_model()
        pl_module = TextSummarization(model=module, tokenizer=tokenizer)
        datamodule = TextSummarizationDataModule(data_source=self.get_data_source(), tokenizer=tokenizer)
        trainer = L.Trainer(**self.get_trainer_settings())

        self._pl_module = pl_module
        self._trainer = trainer

        trainer.fit(pl_module, datamodule)

            for item in tqdm(items):
                abs_fpath = os.path.join(root, name)
      
        print("Uploading checkpoints and logs...")
        for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                abs_fpath = os.path.join(root, name)
                print(abs_fpath)
                self.drive.put(abs_fpath)
        
