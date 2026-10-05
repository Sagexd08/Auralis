import json
import pytest
import torch
from auralis_stt import AuralisSTT, ModelConfig, Tokenizer
from auralis_stt.checkpoint import load_model
from auralis_stt.merge import average_checkpoints
from safetensors.torch import save_file

TEXTS = ["hello world", "good morning everyone"]


def write_checkpoint(path, seed, tokenizer, cfg):
    torch.manual_seed(seed)
    model = AuralisSTT(cfg)
    path.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    tokenizer.save(path / "tokenizer.json")
    return model


def test_average_is_weighted_mean_and_loads_as_a_model(tmp_path):
    tok = Tokenizer.train(TEXTS, 40)
    cfg = ModelConfig(vocab_size=len(tok), d_model=16, n_layers=1, n_heads=2, conv_kernel=7, dropout=0.0)
    a = write_checkpoint(tmp_path / "a", 1, tok, cfg)
    b = write_checkpoint(tmp_path / "b", 2, tok, cfg)
    out = average_checkpoints([tmp_path / "a", tmp_path / "b"], tmp_path / "m", weights=[3, 1])
    merged, merged_tok = load_model(out)
    assert len(merged_tok) == len(tok)
    for (name, expected_a), expected_b in zip(a.state_dict().items(), b.state_dict().values()):
        if expected_a.is_floating_point():
            assert torch.allclose(merged.state_dict()[name], 0.75 * expected_a + 0.25 * expected_b, atol=1e-6)


def test_merge_rejects_mismatched_configs_and_single_checkpoint(tmp_path):
    tok = Tokenizer.train(TEXTS, 40)
    small = ModelConfig(vocab_size=len(tok), d_model=16, n_layers=1, n_heads=2, conv_kernel=7)
    large = ModelConfig(vocab_size=len(tok), d_model=32, n_layers=1, n_heads=2, conv_kernel=7)
    write_checkpoint(tmp_path / "a", 1, tok, small)
    write_checkpoint(tmp_path / "b", 2, tok, large)
    with pytest.raises(ValueError):
        average_checkpoints([tmp_path / "a", tmp_path / "b"], tmp_path / "m")
    with pytest.raises(ValueError):
        average_checkpoints([tmp_path / "a"], tmp_path / "m")
