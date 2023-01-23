import os.path
import string
import random

import torch
import pandas as pd

from lai_tldr.data import (
    SummarizationDataset,
    TLDRDataModule,
)


class BoringTokenizer:
    def __call__(self, texts, max_length: int = 256, **kwargs):
        return {
            "input_ids": torch.rand(max_length),
            "attention_mask": torch.rand(max_length),
        }


def test_summarization_dataset():
    data = {"source_text": ["a", "b", "c"], "target_text": ["d", "e", "f"]}

    dset = SummarizationDataset(
        data=pd.DataFrame(data),
        tokenizer=BoringTokenizer(),
        source_max_token_len=345,
        target_max_token_len=345,
    )

    counter = 0
    for sample in dset:

        assert isinstance(sample, dict)
        keys = list(sample.keys())

        expected_keys = (
            "source_text_input_ids",
            "labels",
            "source_text_attention_mask",
            "labels_attention_mask",
        )
        assert len(keys) == len(expected_keys)

        for k in expected_keys:
            assert k in keys
            assert isinstance(sample[k], torch.Tensor)
            assert sample[k].shape == (345,)

        counter += 1

    assert counter == len(dset) == 3


def test_textsummarization_datamodule(tmpdir):
    printable = string.ascii_lowercase + string.ascii_uppercase + string.ascii_letters
    data = {
        "source_text": [
            "".join(random.choice(printable) for i in range(10)) for _ in range(100)
        ],
        "target_text": [
            "".join(random.choice(printable) for i in range(10)) for _ in range(100)
        ],
    }
    pd.DataFrame(data).to_csv(os.path.join(tmpdir, "data.csv"))

    dm = TLDRDataModule(
        os.path.join(tmpdir, "data.csv"),
        tokenizer=BoringTokenizer(),
    )

    assert dm.train_df is None
    assert dm.val_df is None
    assert dm.test_df is None
    assert dm.train_dataset is None
    assert dm.val_dataset is None
    assert dm.test_dataset is None

    dm.prepare_data()
    assert os.path.exists(os.path.join(tmpdir, "data.csv"))

    dm.setup()

    assert dm.train_df is not None
    assert dm.train_dataset is not None
    assert isinstance(dm.train_df, pd.DataFrame)
    assert isinstance(dm.train_dataset, SummarizationDataset)
    assert len(dm.train_df) == len(dm.train_dataset) == 60

    assert dm.val_df is not None
    assert dm.val_dataset is not None
    assert isinstance(dm.val_df, pd.DataFrame)
    assert isinstance(dm.val_dataset, SummarizationDataset)
    assert len(dm.val_df) == len(dm.val_dataset) == 20

    assert dm.test_df is not None
    assert dm.test_dataset is not None
    assert isinstance(dm.test_df, pd.DataFrame)
    assert isinstance(dm.test_dataset, SummarizationDataset)
    assert len(dm.test_df) == len(dm.test_dataset) == 20

    assert isinstance(dm.train_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(dm.val_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(dm.test_dataloader(), torch.utils.data.DataLoader)
