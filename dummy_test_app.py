#! pip install git+https://github.com/Lightning-AI/LAI-TLDR-Component git+https://github.com/Lightning-AI/lightning-LLMs
#! curl https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv --create-dirs -o ${HOME}/data/summary/news.csv -C -

import lightning
from tests.test_app import DummyTLDR
from lit_llms.tensorboard import MultiNodeLightningTrainerWithTensorboard
app = lightning.LightningApp(
    MultiNodeLightningTrainerWithTensorboard(
        DummyTLDR, num_nodes=2, cloud_compute=lightning.CloudCompute("gpu-fast-multi", disk_size=50),
    )
)
