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
* using multiple GPUs
* across multiple machines
* on your own data
* all without any infrastructure hassle! 

All handled easily with the [Lightning Apps framework](https://lightning.ai/lightning-docs/).

## Run

To run TLDR, paste the following code snippet in a file `app.py`:


```python
# !pip install 'git+https://github.com/Lightning-AI/LAI-TLDR-Component'
import lightning as L
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from lai_tldr import predict, TLDR

sample_text = """
summarize: ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class MyTLDR(TLDR):
    def get_model(self):
        model_type = "t5-base"
        t5 = T5ForConditionalGeneration.from_pretrained(model_type, return_dict=True)
        t5_tokenizer = T5Tokenizer.from_pretrained(model_type)
        return t5, t5_tokenizer

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()
        if from L.app.utilities.cloud import is_running_in_cloud:
            settings["strategy"] = "deepspeed_stage_3_offload"

        settings["precision"] = 16
        return settings

    def get_data_source(self) -> str:
        return "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"

    def run(self):
        super().run()

        # Make a prediction at the end of fine-tuning
        if self._trainer.global_rank == 0:
            predictions = predict(self._pl_module.to(self._trainer.strategy.root_device), sample_text)
            print("Input text:\n", sample_text)
            print("Summarized text:\n", predictions[0])


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTLDR,
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
