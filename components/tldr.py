from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch.nn as nn
import lightning as L
from components.model import TextSummarization, TextSummarizationDataModule, predict

# TODO
Tokenizer = Any

class TLDR(L.LightningWork, ABC):

    @abstractmethod
    def get_model(self) -> Tuple[nn.Module, Tokenizer]:
        """TODO docs"""
        pass

    @abstractmethod
    def get_data_source(self) -> str:
        """TODO docs"""
        pass

    def get_trainer_settings(self):
        """TODO docs"""
        return dict(max_steps=5)  # TODO, model checkpointing etc.

    def run(self):
        module, tokenizer = self.get_model()
        pl_module = TextSummarization(model=module, tokenizer=tokenizer)
        datamodule = TextSummarizationDataModule(data_source=self.get_data_source(), tokenizer=tokenizer)
        trainer = L.Trainer(**self.get_trainer_settings())
        trainer.fit(pl_module, datamodule)

        # TODO: export weights
