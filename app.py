#! pip install pandas
#! pip install scikit-learn

import lightning as L
from lightning.app.components import LightningTrainerMultiNode
from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer
from model import TextSummarization, TextSummarizationDataModule, predict

sample_text = """
ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class TLDR(L.LightningWork):
    def run(self):
        t5 = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

        datamodule = TextSummarizationDataModule(t5_tokenizer)
        model = TextSummarization(model=t5, tokenizer=t5_tokenizer)
        trainer = L.Trainer(max_steps=5)
        trainer.fit(model, datamodule)

        if trainer.global_rank == 0:
            predictions = predict(model.to("cuda"), sample_text)
            print("predictions:", predictions[0])


app = L.LightningApp(
    LightningTrainerMultiNode(
        TLDR,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu"),  # gpu-fast-multi
    )
)
