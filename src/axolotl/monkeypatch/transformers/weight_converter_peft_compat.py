"""Compat shim for peft 0.19.x <-> transformers 5.10.x adapter resume.

peft 0.19.x's ``build_peft_weight_mapping`` reconstructs transformers'
``WeightConverter`` passing ``distributed_operation=`` and
``quantization_operation=`` as ``__init__`` kwargs. transformers 5.10.x's
``WeightConverter.__init__`` only accepts ``(source_patterns, target_patterns,
operations)`` and expects those two to be set as attributes afterward (which is
exactly what transformers' own ``integrations/peft.py`` does). The skew raises
``TypeError: WeightConverter.__init__() got an unexpected keyword argument
'distributed_operation'`` on resume-from-checkpoint adapter loading
(``model.load_adapter`` -> ``set_peft_model_state_dict`` ->
``build_peft_weight_mapping``).

No version combo within axolotl's pins resolves it (peft>=0.19.1,<0.20.0 always
passes the kwargs; transformers 5.10.0-5.10.4 never accept them), so accept the
kwargs and set them as attributes -- matching transformers' own convention.
``WeightTransform.__slots__`` already declares both attributes.
"""

import inspect

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCHED = False


def patch_weight_converter_peft_compat() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import transformers.core_model_loading as core_model_loading
    except ImportError:
        return False

    weight_converter = getattr(core_model_loading, "WeightConverter", None)
    if weight_converter is None:
        return False

    # No-op once upstream's __init__ accepts the kwarg directly.
    if "distributed_operation" in inspect.signature(weight_converter.__init__).parameters:
        _PATCHED = True
        return True

    orig_init = weight_converter.__init__

    def __init__(  # noqa: N807
        self,
        source_patterns,
        target_patterns,
        operations,
        distributed_operation=None,
        quantization_operation=None,
    ):
        orig_init(self, source_patterns, target_patterns, operations)
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    weight_converter.__init__ = __init__
    _PATCHED = True
    LOG.info(
        "Patched transformers WeightConverter.__init__ to accept peft 0.19.x "
        "distributed_operation/quantization_operation kwargs (resume compat)"
    )
    return True
