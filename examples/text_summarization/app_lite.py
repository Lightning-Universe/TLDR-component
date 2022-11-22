#! pip install pandas
#! pip install scikit-learn

import lightning as L
from lightning.app.components import LiteMultiNode
from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer
from model import TextSummarization, TextSummarizationDataModule, predict
from tqdm import tqdm

sample_text = """
ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class HappyTransformer(L.LightningWork):

    def run(self):
        t5 = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

        datamodule = TextSummarizationDataModule(t5_tokenizer)
        model = TextSummarization(model=t5, tokenizer=t5_tokenizer)
        model, lite = self.train(model, datamodule)

        if lite.global_rank == 0:
            predictions = predict(model.to("cuda"), sample_text)
            print("predictions:", predictions[0])
        lite._strategy.barrier()

    def train(self, model, datamodule):
        lite = L.LightningLite()

        if lite.local_rank == 0:
            datamodule.prepare_data()
        datamodule.setup()

        optimizer = model.configure_optimizers()
        model, optimizer = lite.setup(model, optimizer)

        train_dataloader = lite.setup_dataloaders(datamodule.train_dataloader())

        global_step = 0
        max_steps = None
        max_epochs = 10

        for epoch in range(max_epochs):

            for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                loss = model.training_step(batch, batch_idx)
                lite.backward(loss)
                optimizer.step()

                if max_steps is not None and global_step >= max_steps:
                    break

                global_step += 1

            if max_epochs is not None and epoch >= max_epochs:
                break

        return model, lite


app = L.LightningApp(
    LiteMultiNode(
        HappyTransformer,
        num_nodes=4,
        cloud_compute=L.CloudCompute("gpu"),  # gpu-fast-multi
    )
)
