import os
import string
from collections import namedtuple
from random import choice
from time import sleep

import lightning as L
import pandas as pd
import pytest
import torch

from lai_tldr import TLDRDataModule
from lai_tldr.module import TLDRLightningModule

return_type = namedtuple("return_type", ("loss", "logits"))


class BoringModel(torch.nn.Module):
    def __init__(self, target_seq_length: int, vocab_size: int, embed_size: int = 10):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, embed_size)
        self.layer2 = torch.nn.Linear(embed_size, vocab_size)
        self.seq_length = target_seq_length
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None, **kwargs):
        # mimic source_seq_length -> target_seq_length by truncation
        logits = self.layer2(self.wte(input_ids))[:, : self.seq_length, :]
        if labels is None:
            loss = None
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return return_type(loss, logits)

    def generate(self, input_ids, num_beams: int = 1, **kwargs):
        return [self(input_ids).logits.argmax(-1)[0].tolist() for _ in range(num_beams)]


class BoringTokenizer:
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 256,
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "attention_mask": torch.rand(self.max_length),
        }

    def encode(self, *args, **kwargs):
        return self(*args, **kwargs)["input_ids"]

    def decode(self, generated_id, *args, **kwargs):
        return "".join(choice(string.printable) for i in range(generated_id.numel()))


@pytest.mark.parametrize("max_length", [128, 256])
@pytest.mark.parametrize("vocab_size", [10])
def test_module_train(tmpdir, max_length, vocab_size):
    printable = (
        string.ascii_lowercase + string.ascii_uppercase + string.ascii_letters + " "
    )
    data = {
        "source_text": [
            "".join(choice(printable) for i in range(10)) for _ in range(100)
        ],
        "target_text": [
            "".join(choice(printable) for i in range(10)) for _ in range(100)
        ],
    }
    pd.DataFrame(data).to_csv(os.path.join(tmpdir, "data.csv"))

    dm = TLDRDataModule(
        os.path.join(tmpdir, "data.csv"),
        tokenizer=BoringTokenizer(max_length=max_length, vocab_size=vocab_size),
    )

    module = TLDRLightningModule(
        BoringModel(target_seq_length=max_length, vocab_size=vocab_size),
        BoringTokenizer(max_length=max_length, vocab_size=vocab_size),
    )

    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
    )
    trainer.fit(module, dm)
