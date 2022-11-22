from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch.nn as nn
import lightning as L
from llm_components.text_summarization.text_summarization import TextSummarization, TextSummarizationDataModule


class TLDR(L.LightningWork, ABC):
    """Finetune on a text summarization task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.checkpoints = L.app.storage.Drive("lit://checkpoints")

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
            dirpath="checkpoints",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        return dict(max_epochs=5, callbacks=[early_stopping, checkpoints], strategy="ddp_find_unused_parameters=False")

    def run(self):
        module, tokenizer = self.get_model()
        pl_module = TextSummarization(model=module, tokenizer=tokenizer)
        datamodule = TextSummarizationDataModule(data_source=self.get_data_source(), tokenizer=tokenizer)
        trainer = L.Trainer(**self.get_trainer_settings())

        self._pl_module = pl_module
        self._trainer = trainer

        trainer.fit(pl_module, datamodule)

        print("DEBUG: putting files in drive")
        self.checkpoints.put("./checkpoints")
