# Gemma 4 26B-A4B NVFP4 LoRA — new-machine setup (target: 8× B300)

This document is the single source of truth for resuming this training effort on a
fresh machine. It is written for an intelligent agent (or human) to execute: every
section states *why* a step exists so you can adapt when the new machine differs,
and §7 is an ordered verification gauntlet — do not start the full run until every
gate passes. Exact package pins live in `requirements-freeze.txt` next to this file.

---

## 1. What this run is

LoRA fine-tune of **`nvidia/Gemma-4-26B-A4B-NVFP4`** (26B-total / ~4B-active MoE,
NVFP4-quantized base) on **`spellbound-eng/5-12-o-dataset`** (chat-template,
assistant-only targets, last-turn-only), with:

- `prompt_loss_weight: 0.1` — custom feature: prompt tokens contribute 10% loss weight
- 75,000-token packed sequences (`sample_packing: true`), micro-batch 2
- LoRA r=128 / α=256 on text-backbone attention+MLP projections **and** the 3D MoE
  expert parameters (`lora_target_parameters`), fused LoRA kernels on the non-expert path
- ScatterMoE experts with the fused NVFP4 4-bit grouped GEMM (`dsv4_fp4_grouped_mode: nvfp4`)
- Cut Cross-Entropy (CCE) loss, Liger kernels
- FA4 (flash-attention CuTe DSL) on the sliding-window layers via a **required fork**
  (see §2), SDPA on the global layers (`gemma4_hybrid_attn_impl: true`)

**History, condensed** (explains the odd fossils you will encounter):
1. Started on 8× RTX PRO 6000 Blackwell (sm_120). The config file's *header comments*
   still describe that machine — they are stale; trust the config keys, not the prose.
2. FA4 has no SM120 kernels. An attempt to retrofit SM120 support was judged
   infeasible and fully reverted — it exists nowhere in git history. Do not resurrect it.
3. Moved to a single B300 (sm_103). There FA4's dedicated SM100 head_dim=256 kernel
   asserted on Gemma 4's sliding-window layers; we implemented sliding-window support
   in a fork and validated it exhaustively (441/441 upstream tests).
4. The last training attempt on that box crashed with an **unresolved** illegal memory
   access in the ScatterMoE LoRA Triton path (§10). The box was torn down before a
   clean retest. This is the first thing to disambiguate on the new machine.

Known-good reference points from 2-step real-config runs on the single B300:
loss ≈ 3.81 at step 1, **78.3 GiB** VRAM at micro_batch_size=2, `plw=0.0` bit-matches
the no-PLW baseline, PLW costs ~2% tok/s. `grad_norm` logs `inf` on NVFP4 runs
*including the baseline* — pre-existing cosmetic quirk, not a red flag.

---

## 2. Source of truth — repos and exact revisions

| Repo | Branch / rev | What it is |
|---|---|---|
| `github.com/selalipop/axolotl` | `plw-gemma4-26b` @ `476b4267a311b7d629dd6115adb673397d8a698c` | Training framework + all our changes |
| `github.com/selalipop/flash-attention` | `hd256-local-attention` @ `2bbfd4adbdd4b8d8ac4d1486d7525ffc14e4e359` | FA4 fork — **mandatory**, upstream cannot run this model |
| `github.com/axolotl-ai-cloud/ml-cross-entropy` | `5f0c7a7778b5b17d37738fae10065ed3034373af` | CCE fork (not ours; pip-install at this pin) |

### axolotl branch contents (base: upstream `axolotl-ai-cloud/axolotl` main @ `122f13af`)

- `7ca5a518` — **prompt_loss_weight feature** (+ unit/e2e tests). Design invariants
  that are easy to break later:
  - Weights are derived at loss time from `labels`/`input_ids`/`position_ids` only —
    **never** `attention_mask` (it is deleted for gemma4 packing in `compute_loss`).
    Boundary rule: targets at `position_ids==0` get weight 0; padding excluded via
    `pad_token_id`.
  - CCE path: axolotl-side rebind of the CCE fork's `apply_lce` per loaded
    `cut_cross_entropy.transformers.*` module; weights hand off via a module stash in
    `src/axolotl/monkeypatch/loss/prompt_loss_weight.py`. The fork's
    `reduction="none"` NLL is **shift-sliced** `[B, S-1]` (loss for target j+1 at
    index j) — weights must be sliced to match.
  - `num_items_in_batch` becomes the weighted float count; `model_accepts_loss_kwargs`
    is forced True while PLW is active.
  - The multimodal gate keys on processor presence, **not** `cfg.is_multimodal`
    (which is model-family-derived and would wrongly block text-only Gemma 4).
- `f446fe47` — **gemma4 fused-attn hub-kernel alias fix.** When the FA2 package is
  absent, transformers rewrites `_attn_implementation` from `flash_attention_2` to
  `kernels-community/flash-attn2`; the gemma4 fused-attention monkeypatch maps that
  alias back to the canonical name so attention resolves to the FA4-patched
  `flash_attention_2` function instead of the hub kernel. Without this, FA4 is
  silently bypassed (or the hub kernel fails).
- `3ab4daba` — **`exclude_torchao_params_from_ddp_sync` helper** in
  `src/axolotl/utils/quantization.py`. DDP's init-time module-state broadcast
  coalesces with `aten.cat`, which torchao subclasses (`NVFP4Tensor`, `MXTensor`)
  don't implement; the frozen quantized weights load identically on every rank and
  need no sync. **⚠ The helper is defined but NOT yet called anywhere** — see §9.11.
- `476b4267` — `local-run/`: the training config, launch script, this guide, and the
  environment freeze, carried off the old box.

### flash-attention branch contents (base: upstream `Dao-AILab/flash-attention` main @ `002cce0`)

- `d17d137` — **SM100 hd256 sliding-window (local) attention** for the dedicated
  2-CTA kernels that FA4 auto-selects for head_dim=256 on `arch//10 ∈ {10, 11}`.
  Upstream asserts `SM100 forward with head_dim=256 does not support local attention
  yet`; Gemma 4's 26/30 layers are sliding-window at head_dim 256, so this fork is a
  hard requirement. Real bugs fixed beyond lifting the asserts (relevant if rebasing):
  the hd256 *backward* takes window sizes at **constructor/compile time** and its
  compile key must include window *values* (not presence); the `-1` "unbounded"
  sentinel broke one-sided windows in the dK/dV kernel; negative windows (legal:
  bounds tighter than the diagonal) were wrongly squashed to None; the dQ kernel was
  missing OOB masking on the residual K tile (varlen corruption). Validated: custom
  11-case fwd+bwd suite + full upstream `tests/cute` sweep, 441/441 passed.
- `2bbfd4a` — **`nvvm.atomicrmw` compatibility with the pinned
  `nvidia-cutlass-dsl==4.6.0.dev0`** (the `res=` kwarg no longer exists; affects
  `atomic_add_fp32` used by `flash_bwd_mla_sm100.py` / `copy_utils.py`). Any machine
  running this branch with the pinned DSL needs this — do not run from `d17d137`.

### How FA4 actually gets used (no config flag!)

Upstream axolotl's `patch_manager` calls
`src/axolotl/monkeypatch/attention/flash_attn_4.py:patch_flash_attn_4`, which
redirects transformers' FA2 lazy imports to `flash_attn.cute` **automatically** when
(a) `flash_attn.cute` imports successfully and (b) GPU capability major ∈ {9, 10, 11}
(B300 is sm_103 → major 10 ✓). Consequences:

- Installing the FA4 fork **is** the enable switch. The config keeps
  `attn_implementation: flash_attention_2`.
- The FA2 package (`flash-attn` / `flash_attn`) must **NOT** be installed: its regular
  `flash_attn` package shadows the `flash_attn.cute` namespace and breaks the import,
  which silently disables FA4 (the patch just logs and falls back).

---

## 3. Hardware and system assumptions

- 8× B300 (sm_103, ~275 GB usable HBM each). Everything here also works on 1 GPU
  (validated there); §8 covers the 8-GPU deltas.
- NVIDIA driver whose `nvidia-smi` reports **CUDA Version ≥ 13.0** (torch 2.12.1 is a
  CUDA 13.0 build). No system CUDA toolkit is needed — and a system CUDA < 13 is
  actively harmful if it wins on PATH (§6): CUDA 12.8's nvcc cannot target
  `compute_103a`, which the Marlin NVFP4 JIT needs.
- Disk: ≥ 300 GB free. Budget: model 18 GB, raw dataset ~8 GB, tokenized cache ~68 GB,
  outputs ~25 GB+/run (8 saves/epoch of LoRA + trainer state), plus wandb/logs.
- CPU: the old box had 240 cores and used `dataset_num_proc: 92` for the one-time
  tokenization. Scale that key to roughly `cores - 8`.
- Access: an HF token with read access to the `spellbound-eng` org (the dataset is
  org-private; the base model is public), and a wandb login for project
  `gemma4-26b-a4b-spellbound`.

---

## 4. System packages

```bash
apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential git tmux
```

- `python3.12-dev` is load-bearing: Triton kernel builds fail without `Python.h`.
- `ninja` (pip, included in the freeze) is required for the scattermoe/Marlin C++
  extension builds.

---

## 5. Python environment

Two paths; **Path A is strongly preferred** — it reproduces the exact known-good
resolution and sidesteps a day of version whack-a-mole.

### Path A — exact reproduction from the freeze

```bash
python3.12 -m venv /root/.venv
export PATH="/root/.venv/bin:$PATH"
pip install --upgrade pip

git clone https://github.com/selalipop/axolotl.git /root/axolotl
cd /root/axolotl && git checkout plw-gemma4-26b

git clone https://github.com/selalipop/flash-attention.git /root/flash-attention
cd /root/flash-attention && git checkout hd256-local-attention

pip install -r /root/axolotl/local-run/requirements-freeze.txt
pip install -e /root/axolotl --no-deps
pip install -e /root/flash-attention/flash_attn/cute --no-deps
```

Notes on the freeze file:
- It deliberately **excludes** `flash_attn` (FA2) — never install it (§2). If
  anything drags it in later (`pip install` of something with a `flash-attn` dep),
  `pip uninstall -y flash-attn` and re-verify §7.2.
- It excludes the two editables (axolotl, flash-attn-4) — hence the two `-e ... --no-deps`
  installs afterwards. `--no-deps` matters: their deps are already in the freeze, and
  FA4's pyproject would otherwise re-resolve things.
- `cut-cross-entropy` is pinned as a git URL at the required fork commit. Axolotl's
  CCE integration and the PLW rebind are written against exactly this revision
  (its patches align with transformers 5.12.1).
- The nvcc toolchain pins are in there and are a coherent set — see §6.

### Path B — fresh resolve (only if Path A can't apply, e.g. different Python)

`pip install -e /root/axolotl` normally, then force the pins that actually matter,
newest-compatible for the rest:

| Package | Version | Why pinned |
|---|---|---|
| torch / torchvision | 2.12.1 / 0.27.1 | CUDA 13.0 build; torchvision needed by the gemma4 processor import |
| transformers | 5.12.1 | CCE fork patches and axolotl monkeypatches align to it |
| nvidia-cutlass-dsl | 4.6.0.dev0 | FA4's hard pin; the `2bbfd4a` fix targets its API |
| quack-kernels / apache-tvm-ffi / torch-c-dlpack-ext | 0.5.3 / 0.1.12 / 0.1.5 | FA4 runtime deps (not auto-installed due to `--no-deps`) |
| triton | 3.7.1 | scattermoe kernels; the open IMA (§10) was observed here — keep constant to debug apples-to-apples |
| torchao | 0.17.0 | NVFP4 tensor subclass + `adamw_torch_8bit` optimizer |
| accelerate / trl / peft / datasets | 1.13.0 / 1.7.0 / 0.19.1 / 4.8.5 | Known-good set with axolotl base `122f13af` |
| liger-kernel / kernels | 0.8.0 / 0.13.0 | Liger plugin; `kernels` lib is in the attn fallback path |
| nvidia-cuda-nvcc/-crt/nvidia-nvvm | 13.0.88, nvidia-cuda-cccl 13.0.85 | See §6 — unpinned nvcc grabs 13.3 and breaks against CCCL |

Then CCE and the FA4 fork exactly as in Path A. Verify `python -c "import torch;
print(torch.version.cuda)"` → `13.0`; if PyPI's default torch build isn't cu13 on the
new box, use `--index-url https://download.pytorch.org/whl/cu130`.

---

## 6. Runtime environment — every export, and why

`local-run/launch-train.sh` encodes these; adapt its absolute paths if your layout
differs. The exports, with rationale:

```bash
# 1. The venv must win. The old box had a stray /root/.venv-hf-wandb/bin and a dead
#    /snap/bin/accelerate shadowing the real ones. On any new box, verify with:
#    which -a python axolotl accelerate
export PATH="/root/.venv/bin:$PATH"

# 2. The NVFP4 MoE path JIT-builds a Marlin kernel (axolotl_marlin_w4a16_sm103) at
#    first run. It must compile with torch's own CUDA (13.0), not a system CUDA.
#    The venv ships the toolkit under nvidia/cu13 — pinned so nvcc(13.0.88) and
#    CCCL(13.0.85) headers agree; an unpinned nvcc pulls 13.3 and errors with
#    "CUDA compiler and CUDA toolkit headers are incompatible".
export CUDA_HOME="/root/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"

# 3. FA4 kernels are JIT-compiled per (shape-class, flags) on first use — minutes of
#    apparent stall at step 1. The disk cache makes that a one-time cost across
#    restarts. Cache lives at /tmp/$USER/flash_attention_cute_dsl_cache; delete it
#    whenever you change FA4 source (it may not key on source content).
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

export PYTHONUNBUFFERED=1
```

Auth (once per box): `huggingface-cli login` with a token that can read
`spellbound-eng/5-12-o-dataset`, and `wandb login`.

Run inside tmux; the launch script tees each attempt to a timestamped log with a
`latest.log` symlink — keep that pattern, it made crash forensics possible.

---

## 7. Pre-flight verification gauntlet (in order — each gates the next)

**7.1 — GPU + torch**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_capability(0))"
# expect: 2.12.1 13.0 8 (10, 3)
```

**7.2 — FA4 resolves to the fork, FA2 absent**
```bash
python - <<'EOF'
import importlib.metadata as md, os
import flash_attn.cute.interface as fai
print(md.version("flash-attn-4"))                  # 4.0.0b21.dev*+g2bbfd4a*
print(os.path.realpath(fai.__file__))              # must be inside /root/flash-attention
try: md.version("flash-attn"); print("FA2 PRESENT — UNINSTALL IT")
except md.PackageNotFoundError: print("FA2 absent — correct")
EOF
```

**7.3 — FA4 hd256 sliding-window micro-test** (the exact shape class that used to
assert; Gemma-4 sliding layers are head_dim 256, window 512 → `window_size=(511, 0)`
with causal; GQA included). First run JIT-compiles for a few minutes.
```bash
cd /tmp && python - <<'EOF'
import torch
from flash_attn.cute.interface import flash_attn_func
q = torch.randn(2, 4096, 16, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(2, 4096, 8, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(2, 4096, 8, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
out, lse = flash_attn_func(q, k, v, causal=True, window_size=(511, 0))
out.sum().backward()
assert out.isfinite().all() and q.grad.isfinite().all() and k.grad.isfinite().all()
print("FA4 hd256 sliding-window fwd+bwd OK", out.shape)
EOF
```

**7.4 (optional, ~20 min) — upstream numerics tests for the touched paths.** Run from
any directory **other than the repo root** (the repo root's `flash_attn/__init__.py`
imports FA2's missing C extension and shadows the editable install):
```bash
cd /tmp && python -m pytest /root/flash-attention/tests/cute/test_flash_attn.py \
  -q -k "local and 256"
```
If you need specific cases, get exact IDs from `--collect-only -q` — hand-built
parametrize IDs are error-prone. Precedent: the full d=256 local sweep (441 tests)
passed in ~16 min on one B300.

**7.5 — tokenize once, on purpose** (not implicitly under an 8-process launch):
```bash
cd /root && axolotl preprocess /root/axolotl/local-run/26b-a4b-moe-nvfp4-lora.yaml
```
~68 GB cache appears under `./last_run_prepared` (config sets no
`dataset_prepared_path`, so it is CWD-relative — as is `output_dir: ./outputs/...`.
**Always launch from the same working directory**, or add absolute paths to the
config copy). Adjust `dataset_num_proc` to the new box's cores first. If this step
fails with "Messages is null", see §9.8.

**7.6 — 2-step single-GPU smoke.** Copy the config, add `max_steps: 2`, then:
```bash
CUDA_VISIBLE_DEVICES=0 axolotl train <copy>.yaml
```
Expect: a long first step (FA4 JIT + Marlin JIT + CCE/dynamo compile — this also
proves the §6 nvcc setup, since Marlin compiles here), step-1 loss ≈ 3.8, ~78 GiB
peak, `grad_norm` possibly `inf` (known, cosmetic). **This is also the retest for the
open scattermoe IMA (§10)** — on the old box it crashed within the first steps.

**7.7 — 2-step 8-GPU smoke.** Same config copy, no `CUDA_VISIBLE_DEVICES` filter —
`axolotl train` launches accelerate over all visible GPUs. Two things to watch:
- **DDP init**: if it crashes with `NotImplementedError: aten.cat` on
  `NVFP4Tensor`/`MXTensor` during `accelerator.prepare`/model broadcast, wire up the
  prepared fix — §9.11.
- Loss at step 1 should match 7.6 (~3.8); per-GPU VRAM should be roughly the 7.6
  number plus DDP gradient buckets (LoRA-only grads → small).

**7.8 — full run.** Remove `max_steps`, restore/choose `wandb_name` (currently
`gemma4-26b-a4b-plw01`), launch via the script pattern, confirm tok/s and loss curve
on wandb, and that step saves land in `outputs/` (8/epoch + first step).

---

## 8. The config, and what to touch for 8 GPUs

`local-run/26b-a4b-moe-nvfp4-lora.yaml`. Reminder: header prose is stale sm120-era
(§1). Key facts and the deliberate choices:

- **Attention**: `attn_implementation: flash_attention_2` + `gemma4_hybrid_attn_impl:
  true` → FA4 (auto-patched, §2) on sliding-window layers (head_dim 256), SDPA on
  global layers (head_dim 512 — beyond FA4's fwd head-dim support, and fine: only
  4/30 layers).
- **MoE**: `use_scattermoe` + `experts_implementation: scattermoe` +
  `dsv4_fp4_grouped_mode: nvfp4`. On sm_103 the fused grouped GEMM JIT-builds Marlin
  `axolotl_marlin_w4a16_sm103` (the yaml comment claims DeepGEMM on sm100-class —
  observed behavior on sm_103 was Marlin).
- **Loss**: CCE plugin + `prompt_loss_weight: 0.1`. `plw: 0.0` was verified ≡
  baseline; PLW invariants in §2.
- **LoRA**: `lora_dropout` must stay 0 (PEFT ParamWrapper for
  `lora_target_parameters` has no dropout support). The target regex tolerates
  activation-checkpoint wrappers (`_checkpoint_wrapped_module`).
- `ddp_find_unused_parameters: true` is already set (MoE routing + hybrid attention
  can leave params unused in a step).
- `sequence_len: 75000` + `sample_packing` + `excess_length_strategy: drop`;
  `eval_sample_packing: false`.
- Optimizer `adamw_torch_8bit` (torchao 8-bit Adam), cosine LR 2e-4, warmup 10%,
  1 epoch, evals/saves 8 per epoch, `save_first_step: true`.

**8-GPU deltas to consider (decide, don't cargo-cult):**
- Effective batch becomes `micro_batch_size(2) × 8 GPUs × grad_accum(1)` = 16 packed
  75k sequences ≈ up to ~1.2M tokens/step, 8× what the single-GPU smoke validated.
  That is a reasonable batch for a 26B LoRA, and warmup/cosine are ratio-based so the
  schedule adapts — but if you care about matching the validated single-GPU loss
  trajectory, revisit `learning_rate` (or drop `micro_batch_size` to keep tokens/step
  closer, at a throughput cost).
- `dataset_num_proc: 92` → `min(92, cores - 8)`-ish.
- `val_set_size: 0.001` with 8-way DDP eval is fine; leave it.
- Multi-GPU wraps via plain DDP (no deepspeed/FSDP configured, none installed; the
  base is frozen-quantized + LoRA, so DDP is the right tool).

---

## 9. Known failure modes → cause → fix

1. **`AssertionError: SM100 forward with head_dim=256 does not support local
   attention yet`** — you are on upstream/wheel FA4, not the fork. Install the fork
   branch (§5). There is no wheel that works.
2. **`TypeError: ... atomicrmw ... 'res'`** at FA4 JIT time — fork checked out at
   `d17d137` instead of tip `2bbfd4a`, with DSL 4.6.
3. **`ModuleNotFoundError: No module named 'flash_attn.cute'`** — FA2 got installed
   alongside (shadows the namespace), or the editable install is gone. `pip uninstall
   -y flash-attn`, reinstall the editable, re-run §7.2. Also: never run Python with
   CWD = the flash-attention repo root (its `flash_attn/__init__.py` shadows).
4. **FA4 silently not used** (slow sliding-window layers; log line "Flash Attention 4
   is available ... To enable: pip install flash-attn-4") — same causes as (3): the
   auto-patch only fires if `flash_attn.cute` imports.
5. **Marlin JIT fails**: "CUDA compiler and CUDA toolkit headers are incompatible" or
   cannot target `compute_103a` — nvcc/CCCL pin drift or wrong `CUDA_HOME` (§6).
   Standalone repro without a training launch: compile
   `src/axolotl/integrations/kernels/libs/scattermoe_lora/marlin_w4a16/_csrc/libtorch_stable/moe/marlin_moe_wna16/repack_standalone.cu`
   with `$CUDA_HOME/bin/nvcc` targeting `arch=compute_103a,code=sm_103a` (include
   paths from `torch.utils.cpp_extension.include_paths()`).
6. **Triton build fails on `Python.h`** — `apt install python3.12-dev`.
7. **gemma4 processor import error at model load** — torchvision missing or
   mismatched with torch (needs 0.27.1 ↔ 2.12.1).
8. **"Messages is null" during preprocess** — dataset schema drift. The dataset's
   column is `messages` with standard role/content keys; the carried config is
   already correct (`field_messages: messages`, no `message_property_mappings`). If
   it recurs, re-inspect the dataset schema first — it has changed under us once.
9. **`grad_norm: inf` in logs on NVFP4 runs** — pre-existing including at baseline
   (no PLW, no FA4 involvement). Cosmetic as far as observed; loss curves were sane.
10. **Step 1 appears hung for minutes** — FA4 + Marlin + dynamo JIT, worst on a cold
    cache. Enable the FA4 disk cache (§6). Cold total on the old box: single-digit
    minutes.
11. **8-GPU DDP init crash: `NotImplementedError` for `aten.cat` on
    `NVFP4Tensor`/`MXTensor`** — anticipated, prepared-but-unwired fix:
    `exclude_torchao_params_from_ddp_sync(model)` in
    `src/axolotl/utils/quantization.py` (commit `3ab4daba`). It must run on the
    **outermost module DDP will wrap** (after PEFT wrapping — parameter names must
    match what DDP resolves, and PEFT prefixes them with `base_model.model.`), before
    `trainer.train()`. `src/axolotl/integrations/expert_parallel/plugin.py` contains
    a worked example of resolving the ignore-list onto the wrapped model
    (`post_model_build` → re-resolve names). Wire it, test on the 8-GPU smoke (§7.7),
    and commit the wiring to the branch. If DDP init passes without it, note that in
    the commit history and move on — it may only trigger on specific
    torchao/accelerate versions.
12. **Wrong binaries on PATH** (`axolotl: command not found`, mystery accelerate
    versions) — PATH shadowing, §6 item 1.

---

## 10. OPEN ISSUE — ScatterMoE illegal memory access (the reason the last run died)

Last observed crash (single B300, after the FA4 fix unblocked attention):

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  in axolotl/integrations/kernels/libs/scattermoe_lora/
     (grouped_train.py → grouped_lora.py → kernels/ops.py, scatter2scatter path)
```

Status: **unresolved and unreproduced**. A 441-test FA4 sweep was concurrently
holding ~36 GB of the same GPU; an allocation failure inside a custom kernel that
doesn't check its return can surface as an IMA instead of a clean OOM, so
contention is a plausible-but-unconfirmed explanation. The box died before a
clean-GPU retest. The §7.6 smoke run IS the retest.

If it reproduces on an idle GPU, debug it as a real bug in the scatter2scatter
Triton path. Playbook, cheapest-first:
- `CUDA_LAUNCH_BLOCKING=1` to get the true faulting kernel (async IMAs blame the
  wrong op otherwise).
- Bisect the feature, one axis at a time, 2-step runs: `dsv4_fp4_grouped_mode` off
  (separates Marlin grouped GEMM from scattermoe proper) → `use_scattermoe: false`
  (slower per-expert fallback; if this trains, the bug is confined to scattermoe) →
  `prompt_loss_weight` removed (should be irrelevant — loss-side only — but cheap
  to rule out).
- Suspect data-dependence: expert routing at 75k packed sequences (a zero-token
  expert, or an expert-count/offset edge in `scatter2scatter`). Try
  `sequence_len: 4096`-class smoke to see if length correlates.
- `compute-sanitizer --tool memcheck` on a tiny config if it reproduces at small
  scale; at 75k it will be too slow.
- Keep triton at 3.7.1 while debugging (§5 Path B table) so observations stay
  comparable to the original crash.

---

## 11. Deliberately NOT carried over

- The RTX PRO 6000 / SM120 FA4 attempt — reverted, never committed, infeasible
  (FA4 has no SM120 kernel; the SM80-path retrofit was abandoned). The `flash_fwd_sm120.py`
  in the fork is untouched upstream code.
- `outputs/` from the old box — only a step-1 checkpoint existed; restart from scratch.
- `last_run_prepared/` (68 GB tokenized cache) — regenerates in §7.5.
- The old box's venv — it was mutated after the last run (FA4 uninstalled, an FA2
  wheel installed); `requirements-freeze.txt` (captured 2026-07-05, FA2 and
  editables excluded) is the authoritative record instead.

## 12. Quick reference

```bash
# clones
git clone -b plw-gemma4-26b https://github.com/selalipop/axolotl.git /root/axolotl
git clone -b hd256-local-attention https://github.com/selalipop/flash-attention.git /root/flash-attention
# env
python3.12 -m venv /root/.venv && export PATH="/root/.venv/bin:$PATH"
pip install -r /root/axolotl/local-run/requirements-freeze.txt
pip install -e /root/axolotl --no-deps
pip install -e /root/flash-attention/flash_attn/cute --no-deps
# every shell that touches the run
export PATH="/root/.venv/bin:$PATH"
export CUDA_HOME="/root/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
# train (from a stable CWD — cache & outputs are CWD-relative)
cd /root && axolotl train /root/axolotl/local-run/26b-a4b-moe-nvfp4-lora.yaml
```
