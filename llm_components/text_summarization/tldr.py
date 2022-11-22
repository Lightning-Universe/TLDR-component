from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch.nn as nn
import lightning as L
from llm_components.text_summarization.text_summarization import TextSummarization, TextSummarizationDataModule


class TLDR(L.LightningWork, ABC):

    @abstractmethod
    def get_model(self) -> Tuple[nn.Module, Any]:
        """Return your large transformer language model here."""

    @abstractmethod
    def get_data_source(self) -> str:
        """Return a path to a file or a public URL that can be downloaded."""

    def get_trainer_settings(self):
        """Optionally return a dictionary with Lightning Trainer settings."""
        return dict(max_steps=5)  # TODO, model checkpointing etc.

    def run(self):
        module, tokenizer = self.get_model()
        pl_module = TextSummarization(model=module, tokenizer=tokenizer)
        datamodule = TextSummarizationDataModule(data_source=self.get_data_source(), tokenizer=tokenizer)
        trainer = L.Trainer(**self.get_trainer_settings())

        self._pl_module = pl_module
        self._trainer = trainer

        trainer.fit(pl_module, datamodule)

        # TODO: export weights
