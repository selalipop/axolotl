"""End-to-end tests for prompt_loss_weight on the default (non-fused) loss path."""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
)


class TestPromptLossWeight:
    """e2e coverage for prompt_loss_weight without CCE (materialized logits)."""

    @pytest.mark.parametrize("prompt_loss_weight", [0.0, 0.5])
    @pytest.mark.parametrize("sample_packing", [True, False])
    def test_sft_w_prompt_loss_weight(
        self, temp_dir, prompt_loss_weight, sample_packing
    ):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "prompt_loss_weight": prompt_loss_weight,
                "sample_packing": sample_packing,
                "sequence_len": 1024,
                "val_set_size": 0.02,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 5e-4,
                "optimizer": "adamw_torch_fused",
                "output_dir": temp_dir,
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "warmup_steps": 5,
                "bf16": "auto",
                "save_first_step": False,
                "use_tensorboard": True,
                "seed": 42,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        check_tensorboard_loss_decreased(
            temp_dir + "/runs",
            initial_window=5,
            final_window=5,
            max_initial=3.5,
            max_final=3.2,
        )
