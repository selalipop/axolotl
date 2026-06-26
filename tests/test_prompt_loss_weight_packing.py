from types import SimpleNamespace

import torch

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.collators.batching import (
    BatchSamplerDataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)


class _Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    def pad(self, features, **_kwargs):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {}
        for key in features[0]:
            values = []
            for feature in features:
                value = list(feature[key])
                if key == "input_ids":
                    value = value + [self.pad_token_id] * (max_len - len(value))
                values.append(value)
            batch[key] = torch.tensor(values)
        return batch


def test_packed_collators_zero_prompt_loss_weight_at_document_starts():
    packed = [
        [
            {
                "input_ids": [1, 2],
                "labels": [-100, 2],
                "position_ids": [0, 1],
                "loss_weights": [0.1, 1.0],
            },
            {
                "input_ids": [3, 4, 5],
                "labels": [-100, 4, 5],
                "position_ids": [0, 1, 2],
                "loss_weights": [0.1, 1.0, 1.0],
            },
        ]
    ]

    for collator_cls in (
        BatchSamplerDataCollatorForSeq2Seq,
        V2BatchSamplerDataCollatorForSeq2Seq,
    ):
        batch = collator_cls(tokenizer=_Tokenizer())(packed)
        assert batch["loss_weights"].tolist() == [[0.0, 1.0, 0.0, 1.0, 1.0]]


def test_weighted_loss_count_and_labels_ignore_packed_document_starts():
    fake_trainer = SimpleNamespace(
        args=SimpleNamespace(
            average_tokens_across_devices=False,
            n_gpu=1,
            world_size=1,
        ),
        accelerator=SimpleNamespace(parallelism_config=None),
        _loss_shifts_labels=True,
    )
    batch = {
        "loss_weights": torch.tensor([[0.1, 1.0, 1.0, 0.1, 1.0]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
    }

    num_items = AxolotlTrainer._get_num_items_in_batch(
        fake_trainer, [batch], torch.device("cpu")
    )
    assert torch.isclose(num_items, torch.tensor(3.0))

    cleared = AxolotlTrainer._clear_packed_boundary_loss_weights(
        batch["loss_weights"], batch["position_ids"]
    )
    assert cleared.tolist() == [[0.0, 1.0, 1.0, 0.0, 1.0]]

    labels = torch.full((1, 5), -100)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    effective = AxolotlTrainer._make_prompt_weighted_labels(
        input_ids, labels, cleared
    )
    assert effective.tolist() == [[-100, 11, 12, -100, 14]]
