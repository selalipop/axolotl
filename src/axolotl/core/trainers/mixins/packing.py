"""Trainer mixin to support packing"""

from transformers import Trainer


class PackingMixin(Trainer):
    """
    Trainer mixin to support packing
    """

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if (
            self._signature_columns
            and self.args.sample_packing
            and self.args.sample_packing_drop_attention_mask
            and not getattr(self.axolotl_cfg, "prompt_loss_weight", None)
        ):
            set_sig_columns = set(self._signature_columns)
            set_sig_columns.remove("attention_mask")
            self._signature_columns = list(set_sig_columns)
