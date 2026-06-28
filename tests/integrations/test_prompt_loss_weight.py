from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.prompt_loss_weight import PromptLossWeightTrainer
from axolotl.utils.collators.batching import (
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)


class _CaptureModel:
    training = True

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"loss": torch.tensor(0.0)}


class _ToyTokenizer:
    padding_side = "right"

    def pad(
        self,
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        del padding, max_length
        max_len = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of is not None:
            max_len = (
                (max_len + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        output = {}
        for key in features[0]:
            values = []
            for feature in features:
                value = feature[key]
                if hasattr(value, "tolist"):
                    value = value.tolist()
                value = list(value)
                pad_value = -100 if key == "labels" else 0
                values.append(value + [pad_value] * (max_len - len(value)))
            output[key] = torch.tensor(values)
        return output


def _trainer(**args):
    trainer = object.__new__(PromptLossWeightTrainer)
    trainer.axolotl_cfg = SimpleNamespace(prompt_loss_weight=0.1)
    trainer_args = {
        "include_tkps": False,
        "sample_packing": False,
        "sample_packing_drop_attention_mask": False,
        "average_tokens_across_devices": False,
        "n_gpu": 1,
    }
    trainer_args.update(args)
    trainer.args = SimpleNamespace(**trainer_args)
    trainer.state = SimpleNamespace()
    trainer.model_accepts_loss_kwargs = True
    return trainer


def test_collator_adds_loss_attention_mask_for_padded_batch():
    collator = DataCollatorForSeq2Seq(
        _ToyTokenizer(),
        include_loss_attention_mask=True,
    )

    batch = collator(
        [
            {
                "input_ids": [10, 11, 12],
                "labels": [-100, 11, 12],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [20, 21],
                "labels": [-100, 21],
                "attention_mask": [1, 1],
            },
        ]
    )

    torch.testing.assert_close(
        batch["loss_attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )


def test_v2_packed_collator_preserves_loss_attention_mask():
    collator = V2BatchSamplerDataCollatorForSeq2Seq(
        _ToyTokenizer(),
        include_loss_attention_mask=True,
    )

    batch = collator(
        [
            [
                {
                    "input_ids": [10, 11],
                    "labels": [-100, 11],
                    "attention_mask": [1, 1],
                    "position_ids": [0, 1],
                },
                {
                    "input_ids": [20, 21, 22],
                    "labels": [-100, 21, 22],
                    "attention_mask": [1, 1, 1],
                    "position_ids": [0, 1, 2],
                },
            ]
        ]
    )

    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 2, 2, 2]]))
    torch.testing.assert_close(
        batch["loss_attention_mask"],
        torch.tensor([[1, 1, 2, 2, 2]]),
    )


def test_prompt_loss_weight_trainer_fills_active_prompt_labels_only():
    trainer = _trainer()
    model = _CaptureModel()
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "labels": torch.tensor([[-100, -100, 12, 13, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    loss = trainer.compute_loss(model, inputs)

    assert loss.item() == 0.0
    torch.testing.assert_close(
        model.kwargs["labels"],
        torch.tensor([[10, 11, 12, 13, -100]]),
    )
    torch.testing.assert_close(
        model.kwargs["loss_weights"],
        torch.tensor([[0.1, 0.1, 1.0, 1.0, 0.0]], dtype=torch.float32),
    )


def test_prompt_loss_weight_supports_unpadded_batches_without_attention_mask():
    trainer = _trainer()
    model = _CaptureModel()
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "labels": torch.tensor([[-100, 11, -100]]),
    }

    trainer.compute_loss(model, inputs)

    torch.testing.assert_close(model.kwargs["labels"], torch.tensor([[10, 11, 12]]))
    torch.testing.assert_close(
        model.kwargs["loss_weights"],
        torch.tensor([[0.1, 1.0, 0.1]], dtype=torch.float32),
    )


def test_prompt_loss_weight_uses_loss_mask_and_strips_packed_attention_mask():
    trainer = _trainer(sample_packing=True, sample_packing_drop_attention_mask=True)
    model = _CaptureModel()
    denominator = torch.tensor(2.1)
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "labels": torch.tensor([[-100, -100, 12, 13, -100]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 0]]),
        "loss_attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
    }

    trainer.compute_loss(model, inputs, num_items_in_batch=denominator)

    assert "attention_mask" not in model.kwargs
    assert "loss_attention_mask" not in model.kwargs
    # CCE self-normalizes on loss_weights; num_items_in_batch is not forwarded to the model.
    assert "num_items_in_batch" not in model.kwargs
    torch.testing.assert_close(
        model.kwargs["labels"],
        torch.tensor([[10, 11, 12, 13, -100]]),
    )
    torch.testing.assert_close(
        model.kwargs["loss_weights"],
        torch.tensor([[0.1, 0.1, 1.0, 1.0, 0.0]], dtype=torch.float32),
    )


def test_cut_cross_entropy_weighted_loss_shifts_and_masks_boundaries(monkeypatch):
    from cut_cross_entropy.transformers import utils as cce_utils

    captured = {}

    def fake_linear_cross_entropy(e, c, labels, **kwargs):
        del c, labels
        captured["reduction"] = kwargs["reduction"]
        return torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=e.device)

    monkeypatch.setattr(cce_utils, "linear_cross_entropy", fake_linear_cross_entropy)

    opts = SimpleNamespace(to_kwargs=lambda: {"reduction": "mean"})
    loss = cce_utils.apply_lce(
        torch.zeros(1, 5, 2),
        torch.zeros(3, 2),
        torch.tensor([[1, 2, 3, 4, 5]]),
        opts,
        loss_weights=torch.tensor([[0.0, 0.1, 1.0, 1.0, 0.0]]),
        position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
    )

    assert captured["reduction"] == "none"
    assert loss.item() == pytest.approx(21.0 / 1.1)


def test_cut_cross_entropy_uses_external_weighted_denominator(monkeypatch):
    from cut_cross_entropy.transformers import utils as cce_utils

    def fake_linear_cross_entropy(e, c, labels, **kwargs):
        del c, labels, kwargs
        return torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=e.device)

    monkeypatch.setattr(cce_utils, "linear_cross_entropy", fake_linear_cross_entropy)

    opts = SimpleNamespace(to_kwargs=lambda: {"reduction": "mean"})
    loss = cce_utils.apply_lce(
        torch.zeros(1, 5, 2),
        torch.zeros(3, 2),
        torch.tensor([[1, 2, 3, 4, 5]]),
        opts,
        loss_weights=torch.tensor([[0.0, 0.1, 1.0, 1.0, 0.0]]),
        position_ids=torch.tensor([0, 1, 2, 0, 1]),
        num_items_in_batch=torch.tensor(2.0),
    )

    assert loss.item() == pytest.approx(21.0 / 2.0)
