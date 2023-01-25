import os
import string
from collections import namedtuple
from random import choice
from time import sleep

import pandas as pd
import pytest

from lai_tldr import TLDRDataModule
from lai_tldr.module import TLDRLightningModule
import torch
import lightning as L

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
    def __init__(self, max_length: int = 256):
        self.max_length = max_length
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.rand(self.max_length),
            "attention_mask": torch.rand(self.max_length),
        }

    def encode(self, *args, **kwargs):
        return self(*args, **kwargs)["input_ids"]

    def decode(self, generated_id, *args, **kwargs):
        return "".join(
            choice(string.printable) for i in range(generated_id.numel())
        )

@pytest.mark.parametrize('max_length', [128, 256])
def test_module_train(tmpdir, max_length):
    data = {
        "source_text": [
            "".join(choice(string.printable) for i in range(10)) for _ in range(100)
        ],
        "target_text": [
            "".join(choice(string.printable) for i in range(10)) for _ in range(100)
        ],
    }
    pd.DataFrame(data).to_csv(os.path.join(tmpdir, "data.csv"))
    sleep(3)  # makes sure files have been written to disk

    dm = TLDRDataModule(
        os.path.join(tmpdir, "data.csv"),
        tokenizer=BoringTokenizer(max_length=max_length),
    )

    module = TLDRLightningModule(BoringModel(max_length), BoringTokenizer(max_length=max_length))

    trainer = L.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, limit_train_batches=2, limit_val_batches=2, max_epochs=2)
    trainer.fit(module, dm)
