"""Prompt-token loss weighting for Axolotl SFT runs.

The dataset keeps Axolotl's normal assistant-token labels. During loss
calculation this trainer fills masked prompt/input tokens from input_ids and
passes per-token weights to the CCE-patched model forward. Assistant/completion
tokens keep weight 1.0; prompt/input tokens use cfg.prompt_loss_weight.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_distributed

IGNORE_INDEX = -100
LOSS_ATTENTION_MASK = "loss_attention_mask"


def _prompt_loss_weight(cfg) -> float | None:
    prompt_weight = getattr(cfg, "prompt_loss_weight", None)
    if not prompt_weight or prompt_weight <= 0:
        return None
    return float(prompt_weight)


def _active_mask(inputs: dict, reference: torch.Tensor) -> torch.Tensor:
    attention_mask = inputs.get(LOSS_ATTENTION_MASK)
    if attention_mask is None:
        attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)

    mask = attention_mask.to(device=reference.device) != 0
    if mask.shape != reference.shape:
        try:
            mask = torch.broadcast_to(mask, reference.shape)
        except RuntimeError as exc:
            raise ValueError(
                "loss attention mask must broadcast to input/label shape. "
                f"Got mask={tuple(mask.shape)} and reference={tuple(reference.shape)}."
            ) from exc
    return mask


def _build_weighted_labels_and_loss_weights(
    inputs: dict,
    prompt_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = inputs["labels"]
    input_ids = inputs["input_ids"]
    active = _active_mask(inputs, input_ids)

    target_mask = (labels != IGNORE_INDEX) & active
    prompt_mask = (labels == IGNORE_INDEX) & active

    weighted_labels = labels.clone()
    weighted_labels[prompt_mask] = input_ids[prompt_mask]

    loss_weights = torch.zeros_like(weighted_labels, dtype=torch.float32)
    loss_weights[target_mask] = 1.0
    loss_weights[prompt_mask] = prompt_weight

    return weighted_labels, loss_weights, active


class PromptLossWeightArgs(BaseModel):
    """Config arguments for prompt-token loss weighting."""

    prompt_loss_weight: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "If set above zero, include active prompt/input tokens in the loss at "
            "this relative weight while keeping normal target tokens at weight 1.0."
        ),
    )


class PromptLossWeightTrainer(AxolotlTrainer):
    """Axolotl trainer that passes prompt-token loss weights to CCE."""

    # Normalization is left to CCE's apply_lce: given loss_weights it computes the
    # shifted, boundary-masked weighted denominator and clamps it (clamp_min(1.0)),
    # which is exactly the per-step denominator we want. We therefore do NOT override
    # _get_num_items_in_batch or forward num_items_in_batch to the model — that would
    # only matter for cross-microbatch normalization (gradient_accumulation_steps > 1).

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_weight = _prompt_loss_weight(self.axolotl_cfg)
        if prompt_weight is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if "labels" not in inputs or "input_ids" not in inputs:
            model_inputs = dict(inputs)
            model_inputs.pop(LOSS_ATTENTION_MASK, None)
            return super().compute_loss(
                model,
                model_inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        weighted_labels, loss_weights, active = _build_weighted_labels_and_loss_weights(
            inputs, prompt_weight
        )

        if self.args.include_tkps and model.training:
            trainable_tokens = (loss_weights > 0).sum()
            total_tokens = active.sum()

            if is_distributed():
                torch.distributed.all_reduce(
                    trainable_tokens, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    total_tokens, op=torch.distributed.ReduceOp.SUM
                )

            if not hasattr(self.state, "tokens"):
                self.state.tokens = {
                    "trainable": torch.zeros(1),
                    "total": torch.zeros(1),
                }

            self.state.tokens["trainable"] = (
                self.state.tokens["trainable"] + trainable_tokens.detach().cpu()
            )
            self.state.tokens["total"] = self.state.tokens["total"] + total_tokens.cpu()
            self.state.tokens["trainable_tokens"] = trainable_tokens.detach().cpu()

        model_inputs = dict(inputs)
        model_inputs.pop(LOSS_ATTENTION_MASK, None)
        if (
            getattr(self.args, "sample_packing", False)
            and getattr(self.args, "sample_packing_drop_attention_mask", False)
            and "position_ids" in model_inputs
        ):
            model_inputs.pop("attention_mask", None)
        model_inputs["labels"] = weighted_labels
        model_inputs["loss_weights"] = loss_weights

        outputs = model(**model_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


class PromptLossWeightPlugin(BasePlugin):
    """Register prompt-token loss weighting."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.prompt_loss_weight.PromptLossWeightArgs"

    def get_trainer_cls(self, cfg: DictDefault):
        if getattr(cfg, "prompt_loss_weight", None):
            return PromptLossWeightTrainer
        return None
