import logging
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
import io
from typing import Union

import lightning
import os

import pytest

from app import TLDR
from lightning.app.utilities.tracer import Tracer
from lit_llms.tensorboard import (
    MultiNodeLightningTrainerWithTensorboard,
)
from lightning.app.runners import MultiProcessRuntime

from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

from lai_tldr import TLDRLightningModule
from tests.test_module import BoringModel
from lightning.app.testing import LightningTestApp

class DummyTLDR(TLDR):
    boring_model=True
    def run(self):
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
        def trainer_pre_fn(trainer, *args, **kwargs):
            kwargs['max_epochs'] = 2
            kwargs['limit_train_batches'] = 2
            kwargs['limit_val_batches'] = 2

            return {}, args, kwargs

        def multinode_pre_fn(multinode, *args, **kwargs):
            kwargs['num_nodes']=2
            return {}, args, kwargs

        def lm_pre_fn(lm, *args, **kwargs):
            args = list(args)

            model: Union[BoringModel, T5ForConditionalGeneration]
            if args:
                model = args[0]
            else:
                model = kwargs.pop('model', None)

            tokenizer: T5Tokenizer
            if len(args) >= 2:
                tokenizer = args[1]
            else:
                tokenizer = kwargs.pop('tokenizer')

            if self.boring_model:
                # target_seq_length to match datamodule defaults
                model = BoringModel(target_seq_length=128, vocab_size=model.config.vocab_size)
            return {}, (), {'model': model, 'tokenizer': tokenizer}

        def t5tokenizer_pre_fn(cls, *args, **kwargs):
            args = list(args)
            if args:
                args[0] = 't5-small'
            else:
                args.append('t5-small')

            kwargs.pop('pretrained_model_name_or_path', None)

            return {}, args, kwargs

        def t5cond_gen_pre_fn(cls, *args, **kwargs):
            """Only used for cloud but still reduces download size locally"""
            args = list(args)
            if args:
                args[0] = 't5-small'
            else:
                args.append('t5-small')

            kwargs.pop('pretrained_model_name_or_path', None)
            kwargs['return_dict'] = True
            return {}, args, kwargs

        tracer = Tracer()
        tracer.add_traced(lightning.Trainer, '__init__', pre_fn=trainer_pre_fn)
        tracer.add_traced(MultiNodeLightningTrainerWithTensorboard, '__init__', pre_fn=multinode_pre_fn)
        tracer.add_traced(T5Tokenizer, 'from_pretrained', pre_fn=t5tokenizer_pre_fn)
        tracer.add_traced(T5ForConditionalGeneration, 'from_pretrained', pre_fn=t5cond_gen_pre_fn)
        tracer.add_traced(TLDRLightningModule, '__init__', pre_fn=lm_pre_fn)

        tracer._instrument()
        ret_val = super().run()
        tracer._restore()
        return ret_val

def assert_logs(logs):
    expected_strings = [
        # don't include values for actual hardware availability as this may depend on environment.
        'GPU available: ',
        'All distributed processes registered.',
        '674 K    Trainable params\n0         Non - trainable params\n674 K    Total params\n2.699   Total estimated model params size(MB)',
        'Epoch 0:',
        '`Trainer.fit` stopped: `max_epochs=2` reached.',
        'Input text:Input text:\n summarize: ML Ops platforms come in many flavors from platforms that train models'
        ]
    for curr_str in expected_strings:
        assert curr_str in logs

# @pytest.mark.skipif(not bool(int(os.environ.get('SLOW_TEST', '0'))), reason='Skipping Slow Test by default')
def test_app_locally():
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    app = lightning.app.LightningApp(
        MultiNodeLightningTrainerWithTensorboard(
            DummyTLDR, num_nodes=2, cloud_compute=lightning.CloudCompute("gpu-fast-multi", disk_size=50),
        )
    )
    runtime = lightning.app.runners.MultiProcessRuntime(app)
    runtime.dispatch(open_ui=False)
    # TODO: find a way to collect stdout and stderr outputs of multiprocessing to assert expected logs
    # logs = ...
    # assert_logs(logs)
