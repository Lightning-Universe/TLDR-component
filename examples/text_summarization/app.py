# !pip install pandas numpy sentencepiece transformers==4.16.2 scikit-learn
# !pip install 'torch>=1.7.0,!=1.8.0'
# !pip install 'lightning>=1.8.2,<1.8.3'
# !pip install 'git+https://github.com/Lightning-AI/LAI-LLM-components@training'
import os
from typing import Any, Tuple
import torch.nn as nn
import lightning as L
from lightning.app.components import LightningTrainerMultiNode
from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer

from llm_components.text_summarization.text_summarization import predict
from llm_components.text_summarization import TLDR

sample_text = """
ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class MyTLDR(TLDR):

    def get_model(self) -> Tuple[nn.Module, Any]:
        t5 = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        return t5, t5_tokenizer

    def get_data_source(self) -> str:
        return "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"

    def run(self):
        super().run()

        # Make a prediction at the end of fine-tuning
        if self._trainer.global_rank == 0:
            predictions = predict(self._pl_module.to("cuda"), sample_text)
            print("Input text:\n", sample_text)
            print("Summarized text:\n", predictions[0])




app = L.LightningApp(
    LightningTrainerMultiNode(
        MyTLDR,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu", disk_size=50),
    )
)
