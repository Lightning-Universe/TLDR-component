# TL;DR - Text Summarization with Lightning

The TLDR component allows you to pre-train or fine-tune a language model for text-summarization, 
with as many parameters as you want (billions), using multiple GPUs, across multiple machines, 
on your own data, and all without any infrastructure hassle!

```python
# !pip install 'git+https://github.com/Lightning-AI/LAI-TLDR'
import lightning as L
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from lai_tldr.text_summarization import predict, TLDR

sample_text = """
Insert a long text here
"""

class MyTLDR(TLDR):

    def get_model(self):
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
    L.app.components.LightningTrainerMultiNode(
        MyTLDR,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu", disk_size=50),
    )
)


```
Paste the above code snippet in a file `app.py` and run it in the cloud:

```commandline
lightning run app app.py --setup --cloud
```

Don't want to use the public cloud? Contact us at `support@lightning.ai` for alpha access to run on your private cluster (BYOC)!
