import random

import torch
from lai_tldr.text_summarization import predict
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lai_tldr.tldr import TLDR
from collections import namedtuple
import string

return_type = namedtuple("return_type", ("loss", "logits"))


class BoringModel(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.layer = torch.nn.Linear(1, max_length)

    def forward(self, *args, **kwargs):
        logits = self.layer(torch.rand(1))
        loss = logits.sum()
        return return_type(loss, logits)

    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)


class BoringTokenizer:
    def __call__(self, texts, max_length: int = 256, **kwargs):
        return {
            "input_ids": torch.rand(max_length),
            "attention_mask": torch.rand(max_length),
        }

    def encode(self, *args, **kwargs):
        return self(*args, **kwargs)["input_ids"]

    def decode(self, generated_id, *args, **kwargs):
        return "".join(
            random.choice(string.printable) for i in range(generated_id.numel())
        )


class MyTldr(TLDR):
    def __init__(self):
        super().__init__()
        self.drive.component_name = "DummyComponent"

    def get_model(self):
        return BoringModel(max_length=256), BoringTokenizer()

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()
        settings.pop("strategy")
        settings["max_epochs"] = 1
        settings["limit_train_batches"] = 2
        settings["limit_val_batches"] = 2
        return settings

    def get_data_source(self) -> str:
        return "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"


def test_class_instantiation():
    MyTldr()


def test_trainer_settings():
    settings = MyTldr().get_trainer_settings()

    assert isinstance(settings["callbacks"][0], EarlyStopping)
    assert isinstance(settings["callbacks"][1], ModelCheckpoint)
    assert len(settings["callbacks"]) == 2

    assert settings["max_epochs"] == 1
    assert settings["limit_train_batches"] == 2
    assert settings["limit_val_batches"] == 2

    Trainer(**settings)


def test_training():
    summ = MyTldr()
    summ.run()
    predict(summ._pl_module, "This is a great Test!")
