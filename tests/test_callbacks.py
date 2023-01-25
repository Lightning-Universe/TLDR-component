from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lai_tldr.callbacks import default_callbacks


def test_default_callbacks():
    cb = default_callbacks()
    assert len(cb) == 2
    assert isinstance(cb[0], EarlyStopping)
    assert cb[0].monitor == 'val_loss'
    assert cb[0].min_delta == 0.0
    assert cb[0].verbose
    assert cb[0].mode == 'min'

    assert isinstance(cb[1], ModelCheckpoint)
    assert cb[1].save_top_k == 3
    assert cb[1].monitor == 'val_loss'
    assert cb[1].mode == 'min'