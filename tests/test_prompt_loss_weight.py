"""Unit tests for prompt_loss_weight weighted-token loss."""

import pytest
import torch
import torch.nn.functional as F
from transformers.loss.loss_utils import ForCausalLMLoss

from axolotl.monkeypatch.loss.prompt_loss_weight import (
    IGNORE_INDEX,
    build_plw_tensors,
    plw_causal_lm_loss,
    plw_num_items,
    pop_plw_weights,
    set_plw_weights,
)

PAD = 0


@pytest.fixture()
def packed_batch():
    # two packed samples + pad remainder; b=BOS, p=prompt, c=completion
    #             [ b   p   p   c   c ][ b   p   c   c ][pad pad]
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 5, 6, 8, 9, PAD, PAD]])
    labels = torch.tensor([[-100, -100, -100, 8, 9, -100, -100, 8, 9, -100, -100]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1]])
    return input_ids, labels, position_ids


def test_build_plw_tensors_packed(packed_batch):
    input_ids, labels, position_ids = packed_batch
    plw_labels, weights = build_plw_tensors(
        labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=0.1
    )
    # boundaries (position_ids == 0) and padding stay 0; prompts 0.1; completions 1.0
    expected = torch.tensor([[0.0, 0.1, 0.1, 1.0, 1.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(weights, expected)
    assert plw_labels.tolist() == [[-100, 6, 7, 8, 9, -100, 6, 8, 9, -100, -100]]


def test_build_plw_tensors_non_packed_padding():
    input_ids = torch.tensor([[5, 6, 7, 8, PAD, PAD], [5, 6, 7, 8, 9, 10]])
    labels = torch.tensor(
        [[-100, -100, 7, 8, -100, -100], [-100, -100, -100, 8, 9, 10]]
    )
    plw_labels, weights = build_plw_tensors(
        labels, input_ids, None, pad_token_id=PAD, prompt_loss_weight=0.5
    )
    expected = torch.tensor(
        [[0.5, 0.5, 1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]]
    )
    assert torch.allclose(weights, expected)
    assert plw_labels[0].tolist() == [5, 6, 7, 8, -100, -100]


def test_build_plw_tensors_pad_eq_eos(packed_batch):
    # eos doubling as pad: masked eos inside a turn is treated as padding (weight 0)
    input_ids = torch.tensor([[5, 6, PAD, 8, 9]])
    labels = torch.tensor([[-100, -100, -100, 8, 9]])
    _, weights = build_plw_tensors(
        labels, input_ids, None, pad_token_id=PAD, prompt_loss_weight=0.5
    )
    assert torch.allclose(weights, torch.tensor([[0.5, 0.5, 0.0, 1.0, 1.0]]))


def test_build_plw_tensors_all_trained():
    input_ids = torch.tensor([[5, 6, 7]])
    labels = input_ids.clone()
    plw_labels, weights = build_plw_tensors(
        labels, input_ids, None, pad_token_id=PAD, prompt_loss_weight=0.7
    )
    assert torch.equal(plw_labels, labels)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_plw_zero_matches_masked_ce(packed_batch):
    input_ids, labels, position_ids = packed_batch
    torch.manual_seed(0)
    vocab_size = 16
    logits = torch.randn(1, labels.size(1), vocab_size)

    _, weights = build_plw_tensors(
        labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=0.0
    )
    num_items = plw_num_items(weights)
    ref = ForCausalLMLoss(
        logits.clone(), labels.clone(), vocab_size, num_items_in_batch=num_items
    )
    got = plw_causal_lm_loss(
        logits.clone(), labels.clone(), weights, num_items_in_batch=num_items
    )
    assert torch.allclose(ref, got, atol=1e-6)


def test_plw_one_matches_full_ce(packed_batch):
    input_ids, labels, position_ids = packed_batch
    torch.manual_seed(0)
    vocab_size = 16
    logits = torch.randn(1, labels.size(1), vocab_size)

    plw_labels, weights = build_plw_tensors(
        labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=1.0
    )
    got = plw_causal_lm_loss(logits.clone(), plw_labels, weights)

    shift_labels = F.pad(plw_labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    ref = F.cross_entropy(
        logits.float().view(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    assert torch.allclose(ref, got, atol=1e-6)


def test_num_items_matches_loss_denominator(packed_batch):
    # loss normalized by plw_num_items must equal the local weighted mean
    input_ids, labels, position_ids = packed_batch
    torch.manual_seed(1)
    logits = torch.randn(1, labels.size(1), 16)
    plw_labels, weights = build_plw_tensors(
        labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=0.25
    )
    with_count = plw_causal_lm_loss(
        logits.clone(), plw_labels, weights, num_items_in_batch=plw_num_items(weights)
    )
    local_mean = plw_causal_lm_loss(logits.clone(), plw_labels, weights)
    assert torch.allclose(with_count, local_mean, atol=1e-6)


def test_plw_between_masked_and_full(packed_batch):
    input_ids, labels, position_ids = packed_batch
    torch.manual_seed(2)
    logits = torch.randn(1, labels.size(1), 16)

    losses = {}
    for plw in (0.0, 0.3, 1.0):
        plw_labels, weights = build_plw_tensors(
            labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=plw
        )
        losses[plw] = plw_causal_lm_loss(logits.clone(), plw_labels, weights).item()
    low, mid, high = losses[0.0], losses[0.3], losses[1.0]
    assert min(low, high) <= mid <= max(low, high)


def test_logits_length_mismatch_raises(packed_batch):
    input_ids, labels, position_ids = packed_batch
    plw_labels, weights = build_plw_tensors(
        labels, input_ids, position_ids, pad_token_id=PAD, prompt_loss_weight=0.5
    )
    logits = torch.randn(1, labels.size(1) - 1, 16)
    with pytest.raises(ValueError, match="prompt_loss_weight"):
        plw_causal_lm_loss(logits, plw_labels, weights)


def test_weights_stash_set_and_clear():
    weights = torch.ones(1, 4)
    set_plw_weights(weights)
    assert pop_plw_weights() is weights
    assert pop_plw_weights() is None
    set_plw_weights(weights)
    set_plw_weights(None)
    assert pop_plw_weights() is None


class TestPromptLossWeightValidation:
    """Config validation gates for prompt_loss_weight."""

    @pytest.fixture(name="plw_cfg")
    def fixture_plw_cfg(self):
        # plain dict so `|` overrides keys (DictDefault.__or__ keeps existing values)
        return {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            "prompt_loss_weight": 0.1,
        }

    @pytest.mark.parametrize(
        "overrides",
        [
            {"rl": "dpo"},
            {"use_eaft": True},
            {"chunked_cross_entropy": True},
            {"liger_fused_linear_cross_entropy": True},
            {"context_parallel_size": 2},
            {"plugins": ["axolotl.integrations.kd.KDPlugin"]},
            {"processor_type": "AutoProcessor"},
        ],
    )
    def test_incompatible_options_raise(self, plw_cfg, overrides):
        from axolotl.utils.config import validate_config
        from axolotl.utils.dict import DictDefault

        with pytest.raises(ValueError, match="prompt_loss_weight"):
            validate_config(DictDefault(plw_cfg | overrides))

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5])
    def test_out_of_range_raises(self, plw_cfg, bad_value):
        from axolotl.utils.config import validate_config
        from axolotl.utils.dict import DictDefault

        with pytest.raises(ValueError):
            validate_config(DictDefault(plw_cfg | {"prompt_loss_weight": bad_value}))

    def test_valid_config_passes(self, plw_cfg):
        from axolotl.utils.config import validate_config
        from axolotl.utils.dict import DictDefault

        cfg = validate_config(DictDefault(plw_cfg))
        assert cfg.prompt_loss_weight == 0.1

    def test_train_on_inputs_warns_but_passes(self, plw_cfg, caplog):
        from axolotl.utils.config import validate_config
        from axolotl.utils.dict import DictDefault

        cfg = validate_config(DictDefault(plw_cfg | {"train_on_inputs": True}))
        assert cfg.prompt_loss_weight == 0.1
