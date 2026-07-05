"""Weighted-token causal LM loss for `prompt_loss_weight` (PLW).

Tokens masked to -100 at tokenization but backed by a real token get weight
`prompt_loss_weight`; trained tokens get weight 1.0; padding and packed-sample
boundary targets get weight 0. The normalization denominator is the sum of
weights, so `prompt_loss_weight: 0` reproduces standard prompt masking and
`prompt_loss_weight: 1` reproduces full-sequence loss.
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

_plw_weights: torch.Tensor | None = None


def set_plw_weights(weights: torch.Tensor | None) -> None:
    """Stash per-token weights for a loss patch that runs inside model.forward."""
    global _plw_weights
    _plw_weights = weights


def pop_plw_weights() -> torch.Tensor | None:
    global _plw_weights
    weights = _plw_weights
    _plw_weights = None
    return weights


def build_plw_tensors(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    pad_token_id: int | None,
    prompt_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (plw_labels, weights), both aligned with ``labels``.

    ``plw_labels`` unmasks prompt tokens (real token id instead of -100) so a
    loss can be computed there; ``weights[i]`` is the weight of token ``i`` as
    a prediction target.
    """
    prompt_mask = labels.eq(IGNORE_INDEX)
    if pad_token_id is not None:
        prompt_mask = prompt_mask & input_ids.ne(pad_token_id)
    if position_ids is not None:
        # A target with position_id 0 starts a packed sample (or the pad
        # remainder): predicting it would cross sample boundaries.
        prompt_mask = prompt_mask & position_ids.ne(0)

    plw_labels = torch.where(prompt_mask, input_ids, labels)
    weights = labels.ne(IGNORE_INDEX).to(torch.float32)
    weights = weights + prompt_loss_weight * prompt_mask.to(torch.float32)
    return plw_labels, weights


def plw_num_items(weights: torch.Tensor, shifts_labels: bool = True) -> torch.Tensor:
    """Weighted token count matching what `plw_causal_lm_loss` sums over."""
    return weights[..., 1:].sum() if shifts_labels else weights.sum()


def plw_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Per-token weighted CE, shifted/upcast exactly like `ForCausalLMLoss`."""
    if logits.size(-2) != labels.size(-1):
        raise ValueError(
            "prompt_loss_weight: logits seq length "
            f"({logits.size(-2)}) does not match labels ({labels.size(-1)})"
        )
    logits = logits.float()

    labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = labels[..., 1:].contiguous()
    weights = F.pad(weights, (0, 1), value=0.0)
    shift_weights = weights[..., 1:].contiguous()

    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)
    shift_weights = shift_weights.view(-1).to(logits.device)

    per_token_loss = F.cross_entropy(
        logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    loss = (per_token_loss * shift_weights).sum()

    if num_items_in_batch is None:
        # Mirrors reduction="mean" semantics (NaN on fully-masked batches, which
        # the eval nanmean patch relies on).
        return loss / shift_weights.sum()
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch
