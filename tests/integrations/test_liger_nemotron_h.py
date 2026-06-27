import pytest

from axolotl.utils.dict import DictDefault


modeling_nemotron_h = pytest.importorskip(
    "transformers.models.nemotron_h.modeling_nemotron_h"
)


@pytest.fixture(autouse=True)
def restore_nemotron_h_rms_norm():
    original = modeling_nemotron_h.NemotronHRMSNorm
    yield
    modeling_nemotron_h.NemotronHRMSNorm = original


def _cfg(**overrides):
    data = {
        "model_config_type": "nemotron_h",
        "liger_rope": False,
        "liger_cross_entropy": False,
        "liger_fused_linear_cross_entropy": False,
        "liger_rms_norm": True,
        "liger_rms_norm_gated": False,
        "liger_layer_norm": False,
        "liger_glu_activation": False,
        "liger_use_token_scaling": False,
        "torch_compile": False,
        "base_model": "fake/nemotron_h",
        "trust_remote_code": False,
    }
    data.update(overrides)
    return DictDefault(data)


def test_nemotron_h_liger_rms_norm_replaces_transformers_class():
    from axolotl.integrations.liger.plugin import LigerPlugin
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    original = modeling_nemotron_h.NemotronHRMSNorm

    LigerPlugin().pre_model_load(_cfg())

    patched = modeling_nemotron_h.NemotronHRMSNorm
    assert patched is not original
    assert issubclass(patched, LigerRMSNorm)

    norm = patched(16, eps=1e-5)
    assert norm.variance_epsilon == pytest.approx(1e-5)
    assert norm.offset == pytest.approx(0.0)
    assert norm.casting_mode == "llama"
    assert norm.in_place is False
    assert norm.elementwise_affine is True


def test_nemotron_h_unsupported_liger_model_kernels_warn(caplog):
    from axolotl.integrations.liger.plugin import LigerPlugin

    caplog.set_level("WARNING", logger="axolotl.integrations.liger.plugin")

    LigerPlugin().pre_model_load(
        _cfg(
            liger_rope=True,
            liger_glu_activation=True,
            liger_layer_norm=True,
            liger_rms_norm_gated=True,
        )
    )

    messages = "\n".join(record.message for record in caplog.records)
    assert "liger_rope is not directly applied for nemotron_h" in messages
    assert "liger_glu_activation is not supported for nemotron_h" in messages
    assert "liger_layer_norm is not supported for nemotron_h" in messages
    assert "liger_rms_norm_gated is not supported for nemotron_h" in messages


def test_nemotron_h_liger_cross_entropy_warns(caplog):
    from axolotl.integrations.liger.plugin import LigerPlugin

    caplog.set_level("WARNING", logger="axolotl.integrations.liger.plugin")

    LigerPlugin().pre_model_load(_cfg(liger_cross_entropy=True))

    messages = "\n".join(record.message for record in caplog.records)
    assert "liger_cross_entropy is not directly supported for nemotron_h" in messages


def test_nemotron_h_liger_flce_uses_generic_patch(monkeypatch):
    from axolotl.integrations.liger.plugin import LigerPlugin
    import trl.trainer
    from trl.experimental.orpo import ORPOTrainer

    trl.trainer.ORPOTrainer = ORPOTrainer

    from axolotl.integrations.liger.models import base

    called = {}

    def record_patch_lce_forward(model_type):
        called["model_type"] = model_type

    monkeypatch.setattr(base, "patch_lce_forward", record_patch_lce_forward)

    LigerPlugin().pre_model_load(
        _cfg(liger_rms_norm=False, liger_fused_linear_cross_entropy=True)
    )

    assert called == {"model_type": "nemotron_h"}


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="Liger RMSNorm uses a CUDA Triton kernel",
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_nemotron_h_liger_rms_norm_matches_original(dtype):
    import torch

    from axolotl.integrations.liger.plugin import LigerPlugin

    torch.manual_seed(0)
    original_cls = modeling_nemotron_h.NemotronHRMSNorm
    original = original_cls(32, eps=1e-5).cuda()

    LigerPlugin().pre_model_load(_cfg())
    patched = modeling_nemotron_h.NemotronHRMSNorm(32, eps=1e-5).cuda()
    patched.weight.data.copy_(original.weight.data)

    torch_dtype = getattr(torch, dtype)
    hidden_states = torch.randn(4, 11, 32, device="cuda", dtype=torch_dtype)

    expected = original(hidden_states)
    actual = patched(hidden_states)

    if torch_dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
