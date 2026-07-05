"""
Simple end-to-end test for Cut Cross Entropy integration
"""

import sys

import pytest
import torch

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils import get_pytorch_version
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
)


@pytest.fixture()
def min_cfg(temp_dir):
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "plugins": [
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],
        "cut_cross_entropy": True,
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
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "optimizer": "adamw_torch_fused",
        "output_dir": temp_dir,
        "lr_scheduler": "cosine",
        "max_steps": 40,
        "warmup_steps": 5,
        "bf16": "auto",
        "save_first_step": False,
        "use_tensorboard": True,
        "seed": 42,
    }


class TestCutCrossEntropyIntegration:
    """
    e2e tests for cut_cross_entropy integration with Axolotl
    """

    def test_llama_w_cce(self, min_cfg, temp_dir):
        cfg = DictDefault(min_cfg)
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=2.2,
                max_final=2.0,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_weighted_apply_lce_matches_reference(self):
        """Fused reduction="none" weighted loss == materialized weighted CE."""
        from cut_cross_entropy.transformers import llama as llama_mod
        from cut_cross_entropy.transformers.utils import PatchOptions

        from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin
        from axolotl.monkeypatch.loss.prompt_loss_weight import (
            build_plw_tensors,
            plw_causal_lm_loss,
            set_plw_weights,
        )

        bindings = {
            name: mod.apply_lce
            for name, mod in sys.modules.items()
            if name.startswith("cut_cross_entropy.transformers")
            and hasattr(mod, "apply_lce")
        }
        try:
            CutCrossEntropyPlugin()._install_plw_apply_lce()
            assert (
                llama_mod.apply_lce
                is not bindings["cut_cross_entropy.transformers.llama"]
            )

            torch.manual_seed(7)
            dev = "cuda:0"
            seq_len, dim, vocab, pad = 256, 128, 4096, 0
            input_ids = torch.randint(1, vocab, (1, seq_len), device=dev)
            position_ids = torch.arange(64, device=dev).repeat(4).unsqueeze(0)
            labels = torch.where(
                (position_ids % 64) < 32,
                torch.full_like(input_ids, -100),
                input_ids,
            )
            input_ids[:, -8:] = pad
            labels[:, -8:] = -100

            plw_labels, weights = build_plw_tensors(
                labels,
                input_ids,
                position_ids,
                pad_token_id=pad,
                prompt_loss_weight=0.3,
            )
            num_items = weights[..., 1:].sum()
            # filtering off for an exact gradient comparison
            opts = PatchOptions(
                impl="cce",
                reduction="mean",
                filter_eps=None,
                accum_e_fp32=False,
                accum_c_fp32=False,
                filter_e_grad=True,
                filter_c_grad=True,
                train_only=False,
            )
            e0 = torch.randn(1, seq_len, dim, device=dev, dtype=torch.bfloat16) * 0.05
            c0 = torch.randn(vocab, dim, device=dev, dtype=torch.bfloat16) * 0.05

            e_f = e0.clone().requires_grad_(True)
            c_f = c0.clone().requires_grad_(True)
            set_plw_weights(weights)
            loss_fused = llama_mod.apply_lce(
                e_f, c_f, plw_labels, opts, num_items_in_batch=num_items
            )
            loss_fused.backward()

            e_r = e0.clone().requires_grad_(True)
            c_r = c0.clone().requires_grad_(True)
            loss_ref = plw_causal_lm_loss(
                e_r.float() @ c_r.float().T,
                plw_labels,
                weights,
                num_items_in_batch=num_items,
            )
            loss_ref.backward()

            assert torch.allclose(loss_fused.float(), loss_ref.float(), rtol=2e-3)
            for grad_fused, grad_ref in ((e_f.grad, e_r.grad), (c_f.grad, c_r.grad)):
                grad_fused, grad_ref = grad_fused.float(), grad_ref.float()
                rel = (grad_fused - grad_ref).norm() / grad_ref.norm().clamp_min(1e-12)
                cos = torch.nn.functional.cosine_similarity(
                    grad_fused.flatten(), grad_ref.flatten(), dim=0
                )
                assert rel < 5e-2 and cos > 0.999, (rel, cos)

            # without stashed weights the wrapper defers to the original
            loss_plain = llama_mod.apply_lce(
                e0.clone(), c0.clone(), labels, opts, num_items_in_batch=num_items
            )
            loss_orig = bindings["cut_cross_entropy.transformers.llama"](
                e0.clone(), c0.clone(), labels, opts, num_items_in_batch=num_items
            )
            assert torch.allclose(loss_plain.float(), loss_orig.float())
        finally:
            set_plw_weights(None)
            for name, fn in bindings.items():
                sys.modules[name].apply_lce = fn

    def test_qwen2_w_cce(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "plugins": [
                    "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
                ],
                "cut_cross_entropy": True,
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
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "optimizer": "adamw_torch_fused",
                "output_dir": temp_dir,
                "lr_scheduler": "cosine",
                "max_steps": 50,
                "bf16": "auto",
                "save_first_step": False,
                "use_tensorboard": True,
                "seed": 42,
            }
        )
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=5.0,
                max_final=4.7,
            )

    def test_llama_w_cce_prompt_loss_weight(self, min_cfg, temp_dir):
        cfg = DictDefault(
            min_cfg
            | {
                "prompt_loss_weight": 0.3,
                "sample_packing": True,
                "micro_batch_size": 2,
            }
        )
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        # weighted loss includes prompt tokens, so it sits above the
        # completion-only thresholds used by the other CCE tests
        check_tensorboard_loss_decreased(
            temp_dir + "/runs",
            initial_window=5,
            final_window=5,
            max_initial=3.5,
            max_final=3.2,
        )

    @pytest.mark.parametrize(
        "attention_type",
        [
            "flash_attention",
            "sdp_attention",
            # "xformers_attention",
        ],
    )
    def test_llama_w_cce_and_attention(self, min_cfg, temp_dir, attention_type):
        cfg = DictDefault(
            min_cfg
            | {
                attention_type: True,
            }
        )
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=2.2,
                max_final=2.0,
            )
