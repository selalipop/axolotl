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

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_weight = getattr(self.axolotl_cfg, "prompt_loss_weight", None)
        if not prompt_weight or prompt_weight <= 0:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if "labels" not in inputs or "input_ids" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        active_mask = (
            attention_mask.to(dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )

        target_mask = labels != -100
        prompt_mask = (~target_mask) & active_mask

        weighted_labels = labels.clone()
        weighted_labels[prompt_mask] = input_ids[prompt_mask]

        loss_weights = torch.zeros_like(weighted_labels, dtype=torch.float32)
        loss_weights[target_mask] = 1.0
        loss_weights[prompt_mask] = float(prompt_weight)

        if self.args.include_tkps and model.training:
            trainable_tokens = (loss_weights > 0).sum()
            total_tokens = input_ids.numel()
            total_tokens = torch.tensor(total_tokens, device=input_ids.device)

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
