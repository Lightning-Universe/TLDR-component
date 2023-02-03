<div align="center">
    <h1>
        <img src="https://lightningaidev.wpengine.com/wp-content/uploads/2022/11/image-6.png">
        <br>
        TL;DR with Lightning
        </br>
    </h1>

<div align="center">

<p align="center">
  <a href="#run">Run</a> •
  <a href="https://www.lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/lightning-docs/">Docs</a>
</p>

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://lightning.ai/lightning-docs/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://www.pytorchlightning.ai/community)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

</div>
</div>

______________________________________________________________________

Use TLDR to pre-train or fine-tune a large language model for text summarization,
with as many parameters as you want (up to billions!).

You can do this:

- using multiple GPUs
- across multiple machines
- on your own data
- all without any infrastructure hassle!

All handled easily with the [Lightning Apps framework](https://lightning.ai/lightning-docs/).

## Run

To run TLDR, paste the following code snippet in a file `app.py`:

```python
# !pip install git+https://github.com/Lightning-AI/LAI-TLDR-Component git+https://github.com/Lightning-AI/lightning-LLMs
# !curl https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv --create-dirs -o ${HOME}/data/summary/news.csv -C -

import lightning as L
import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

from lit_llms.tensorboard import (
    DriveTensorBoardLogger,
    MultiNodeLightningTrainerWithTensorboard,
)

from lai_tldr import TLDRDataModule, default_callbacks, predict, TLDRLightningModule


class TLDR(L.LightningWork):
    """Finetune on a text summarization task."""

    def __init__(self, tb_drive, **kwargs):
        super().__init__(**kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        # for huggingface/transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # --------------------
        # CONFIGURE YOUR MODEL
        # --------------------
        model_type = "t5-base"
        t5_tokenizer = T5Tokenizer.from_pretrained(model_type)
        t5_model = T5ForConditionalGeneration.from_pretrained(
            model_type, return_dict=True
        )

        lightning_module = TLDRLightningModule(t5_model, tokenizer=t5_tokenizer)

        # -------------------
        # CONFIGURE YOUR DATA
        # -------------------
        data_module = TLDRDataModule(
            os.path.expanduser("~/data/summary/news.csv"), t5_tokenizer
        )

        # -----------------
        # RUN YOUR TRAINING
        # -----------------
        strategy = (
            "deepspeed_stage_3_offload"
            if L.app.utilities.cloud.is_running_in_cloud()
            else "ddp"
        )
        trainer = L.Trainer(
            max_epochs=2,
            limit_train_batches=250,
            precision=16,
            strategy=strategy,
            callbacks=default_callbacks(),
            log_every_n_steps=1,
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
        )
        trainer.fit(lightning_module, data_module)

        if trainer.global_rank == 0:
            sample_text = (
                "summarize: ML Ops platforms come in many flavors from platforms that train models to platforms "
                "that label data and auto-retrain models. To build an ML Ops platform requires dozens of "
                "engineers, multiple years and 10+ million in funding. The majority of that work will go into "
                "infrastructure, multi-cloud, user management, consumption models, billing, and much more. "
                "Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want "
                "to enable (label data then train models), Lightning will handle all the infrastructure, billing, "
                "user management, and the other operational headaches."
            )
            predictions = predict(
                lightning_module.to(trainer.strategy.root_device), sample_text
            )
            print("Input text:\n", sample_text)
            print("Summarized text:\n", predictions[0])


app = L.LightningApp(
    MultiNodeLightningTrainerWithTensorboard(
        TLDR,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi", disk_size=50),
    )
)
```

### Running locally

```bash
lightning run app app.py --setup
```

### Running on cloud

```bash
lightning run app app.py --setup --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!
