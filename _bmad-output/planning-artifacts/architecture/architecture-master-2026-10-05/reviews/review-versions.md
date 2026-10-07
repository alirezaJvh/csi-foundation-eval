# Review — versions, existence and compatibility (reality-check lens)

- **Target:** `../ARCHITECTURE-SPINE.md` (draft, updated 2026-10-06) + `../.memlog.md`
- **Reviewer lens:** was every committed technology decision verified against reality (current versions, existence, fit, mutual compatibility) rather than asserted?
- **Date:** 2026-10-06
- **Method:** PyPI JSON API, PyTorch CPU index listing, Hugging Face model API, raw upstream source, GitHub API, plus a hands-on check: a throwaway uv project in the session scratchpad (not in the repo) pinned to the exact Stack table, then `uv lock`, `uv sync`, weight loading and CPU forward passes, and ruff, ty and import-linter runs.

## Verdict

**The spine needs changes before the build starts.** Every named package, version, HF repo and pinned sha exists, and nearly all versions are the latest on PyPI. The memlog records these lookups, so the versions were checked, not taken from memory. **Compatibility between the pins was not checked:**

1. The Stack table as written **cannot be resolved** (mantis-tsfm vs pandas 3).
2. Following AD-15 as written for WavesFM loads the model with a **randomly initialized patch embedding**, and nothing reports it.
3. Several smaller gaps follow from treating vendored code and model loading as simpler than they are.

Once the fixes below are in, the full stack locks, installs CPU-only and runs LWM, WavesFM and Mantis on CPU. That was verified hands-on.

## Summary table

| # | Severity | Item |
| --- | --- | --- |
| F1 | **Blocker** | mantis-tsfm 1.1.0 declares `pandas<3.0`, so the Stack (pandas 3.0.6) is unsatisfiable |
| F2 | **High** | The WavesFM pretrain checkpoint's patch-embed keys don't match `models_vit.py`, and upstream's `strict=False` silently drops them |
| F3 | **High** | Vendored LWM/WavesFM code fails "ruff + ty must pass" and needs import edits. LWM's `input_preprocess.py` pulls in DeepMIMO, umap and matplotlib |
| F4 | Medium | LWM `lwm.from_pretrained` fails on the published weights (`module.` prefix, default `device='cuda'`) |
| F5 | Medium | `torchvision` (a hard dependency of timm) and the uv index wiring for it are missing from the Stack |
| F6 | Medium | Mantis-8M and timm ImageNet weights have no pinned revision, which conflicts with AD-10/AD-13 |
| F7 | Low | uv 0.11.14 is the locally installed version, not the current one (0.12.23) |
| F8 | Low | Pinned HF revisions use 7-char short shas. The cache key and provenance should use the full 40-char sha |
| F9 | Low | The import-linter claim is realistic, but "notebooks import only results_api" can't be enforced by it |
| F10 | Info | Items confirmed OK (versions, Python 3.12 fit, CPU wheels, timm private API, Mantis on CPU, ty/ruff config) |

---

## F1 — Blocker: Stack is unsatisfiable (mantis-tsfm pins pandas < 3)

**Evidence**

```
$ curl -s https://pypi.org/pypi/mantis-tsfm/json  → info.requires_dist
['datasets>=4.0', 'einops<0.9,>=0.8', 'huggingface-hub>=0.23', 'numpy<3.0,>=1.23',
 'pandas<3.0,>=1.5', 'safetensors>=0.4', 'scikit-learn<2.0,>=1.2', 'torch<3.0,>=1.12', 'tqdm<5.0,>=4.64']
```

`uv lock` (uv 0.11.14) on a pyproject with exactly the Stack pins:

```
× No solution found when resolving dependencies:
╰─▶ Because mantis-tsfm==1.1.0 depends on pandas>=1.5,<3.0 and your project
    depends on mantis-tsfm==1.1.0, we can conclude that your project depends
    on pandas>=1.5,<3.0.
    And because your project depends on pandas==3.0.6, ... unsatisfiable.
```

mantis-tsfm's code never imports pandas. Its imports, scanned from the 1.1.0 wheel: `datasets, einops, huggingface_hub, numpy, sklearn, torch, tqdm`. The cap is metadata only. With `[tool.uv] override-dependencies = ["pandas==3.0.6"]` the lock resolves (72 packages) and `uv sync` installs.

mantis-tsfm 1.1.0 (2026-08-17) is the latest release, so upgrading doesn't help. Its other ranges (torch<3, numpy<3, sklearn<2) are compatible with the pins.

**Suggested fix:** pick one and record it in the Stack and the memlog:
- (a) Keep pandas 3.0.6 and add `[tool.uv] override-dependencies = ["pandas==3.0.6"]`, with a comment explaining it ("mantis-tsfm caps pandas<3 but never imports it, verified 2026-10-06").
- (b) Pin `pandas` to the latest 2.x.

(a) is the smaller change. Either way, add a note that **the Stack was verified by an actual `uv lock`**, not by per-package lookups.

## F2 — High: WavesFM pretrained patch embedding is silently dropped

**Evidence.** Loaded `wavesfm-v1p0.pth` at HF `6a63011e…` into `models_vit.vit_multi_small(modality='vision', ...)` from GitHub `main` @ `0279d3b11a` (2026-05-28), with timm 1.0.30:

```
checkpoint key:  vis_patch_embed.0.proj.weight (256, 1, 16, 16)
model expects:   vis_patch_embed.proj.weight
load_state_dict(strict=False) → missing ['head.weight','head.bias',
   'vis_patch_embed.proj.weight','vis_patch_embed.proj.bias']
```

Upstream `main_finetune.py` lines 209–216 and `hub.py` load with `strict=False` and only print the message. Used the same way here, the frozen WavesFM extractor runs with a **randomly initialized vision patch embedding**, and every downstream number would be wrong without any error.

After remapping `vis_patch_embed.0.` → `vis_patch_embed.`, the only missing keys are `head.*`, as expected, and a forward pass on `(2,1,224,224)` works on CPU.

There are two related traps:
- `use_conditional_ln=False` also loads "cleanly" under `strict=False`. The checkpoint's `mod_ln_scale/bias.vision` then appear only as *unexpected* keys and are ignored.
- The HF `config.json` holds only `{"model_type","architectures"}`, with no `model_args`. The constructor arguments (`vit_multi_small`, 1 input channel, 224/16, conditional LN) have to be inferred from shapes: `cls_token (1,1,256)`, `vis_pos_embed (1,197,256)`, `vis_patch_embed.0.proj.weight (256,1,16,16)`.

The spine pins the HF weights sha but **no GitHub commit for the vendored `models_vit.py`/`pos_embed.py`**. AD-15 asks SOURCE.md for a commit sha, but the Stack row lists only the HF sha.

**Suggested fix:**
- Stack row: add the GitHub source pin `AhmedTarek62/wavesfm @ 0279d3b11afd7691f2565a8361a5d0d60b3ae363` for the vendored files.
- AD-15 (or the extractor contract tests): every pretrained extractor loads weights with a **key remap + `strict=True` on encoder keys**. Equivalently, it asserts that `missing_keys` contains only head keys and `unexpected_keys` only decoder/mask keys. A loader that discards pretrained encoder weights is an `ExtractorError`.
- Record the WavesFM constructor arguments (`vit_multi_small`, `vis_in_chans=1`, `vis_img_size=224`, `vis_patch=16`, `use_conditional_ln=True`, `modality='vision'`) and the `vis_patch_embed.0.` remap in the adapter's SOURCE.md.

## F3 — High: vendored code vs "ruff + ty must pass", import edits, and LWM's DeepMIMO coupling

**Evidence**

1. **LWM `input_preprocess.py` is tied to DeepMIMO.** Top-level imports at lines 11–19: `import DeepMIMOv3` and `from utils import generate_gaussian_noise, plot_coverage`. LWM's `utils.py` imports `matplotlib`, `sklearn.decomposition`, `sklearn.manifold` and `umap` at module level. Vendoring the file as-is therefore requires DeepMIMOv3, umap-learn and matplotlib.
   - The 4×4 patching itself, `patch_maker` (lines 259–301), is pure numpy. The `[CLS]` token is just `0.2 * np.ones(patch_size)` (lines 198–201).
   - `make_sample` calls `np.random.seed(seed)` and `np.random.choice` **even when `mask=False`**. That mutates global NumPy RNG state, which conflicts with AD-12 (seeds owned by config and runner).
   - `lwm_model.py` itself imports only `torch` and `numpy`, as the memlog says.
2. **WavesFM `models_vit.py` line 11 is `from pos_embed import ...`**, an absolute top-level import. Inside `adapters/extractors/wavesfm/_vendor/` it raises `ModuleNotFoundError` unless edited to `from .pos_embed import ...`. Line 8–9 imports from timm are fine (see F10). `pos_embed.py` imports only numpy and torch. No other extra dependencies are needed.
3. **The quality gate fails on unmodified vendored code.**
   - `ruff check --select E,F,I,UP,B` on the three vendored files: **32 errors**.
   - `ty check` (0.0.84): **10 diagnostics** in `models_vit.py`, e.g. `unresolved-attribute ... on Tensor in union Tensor | Module` at line 363.
   - With `[tool.ty.src] exclude = [".../_vendor/**"]`, ty correctly skipped vendored code in the scratch project. Ruff still reported 3 errors in a vendored `lwm_model.py` until it is excluded.

**Suggested fix:**
- AD-15: "vendor the model definition plus **only** the pure functions needed (LWM `patch_maker`, the CLS constant), as an excerpt; never `input_preprocess.py` or `utils.py` whole. Reimplement `make_sample` without global RNG calls." Note the `from .pos_embed` edit as an expected local edit in SOURCE.md.
- Code-quality convention: "`_vendor/` is excluded from ruff (`extend-exclude`) and ty (`[tool.ty.src] exclude`); adapters wrapping vendored code are not." Without this, the "ruff + ty must pass" rule and the "copy unchanged + record edits" rule contradict each other.

## F4 — Medium: LWM's own `from_pretrained` doesn't work with its published weights

**Evidence.** `models/model.pth` at `1b68512dbe97…` is an `OrderedDict` whose keys start with `module.` (DataParallel), e.g. `module.decoder_bias`.

- `lwm.from_pretrained(path, device='cpu')` → `RuntimeError: Missing key(s) in state_dict: "decoder_bias", "embedding.proj.weight", ...`
- Its default is also `device='cuda'`.
- Upstream `main.py` lines 79–81 strips the prefix by hand: `{k.replace("module.", ""): v ...}`. After stripping, the strict load succeeds.
- `torch.load` with the torch 2.14 default `weights_only=True` works, since the file is a plain state dict.

**Suggested fix:** in the LWM adapter (or SOURCE.md), load the state dict directly, strip `module.`, use `strict=True` and `map_location='cpu'`. Don't use `lwm.from_pretrained`. The F2 contract test covers this too.

## F5 — Medium: torchvision missing from Stack, and the uv CPU-index wiring is unspecified

**Evidence**

- `timm` 1.0.30 `requires_dist = ['torch','torchvision','pyyaml','huggingface_hub','safetensors']`.
- The PyPI `torch==2.14.1` on Linux depends on `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13`, `triton~=3.8.0` and similar. Resolving any torch-family package from PyPI on Linux brings in the CUDA stack.
- The CPU index has both packages for this platform: `torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` and `torchvision-0.29.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` (https://download.pytorch.org/whl/cpu/torch/, …/torchvision/).
- uv docs (https://docs.astral.sh/uv/guides/integration/pytorch/) recommend `[[tool.uv.index]] name="pytorch-cpu" url=".../whl/cpu" explicit=true` together with `[tool.uv.sources] torch = [{index="pytorch-cpu"}]`, and a source entry for every torch-family package.
- With both torch and torchvision mapped, the lock gave `torch 2.14.1+cpu` and `torchvision 0.29.1+cpu`, with **0 `nvidia-*` packages**. torchvision 0.29.1 requires `torch>=2.14.0`.

**Suggested fix:** add `torchvision 0.29.1 (CPU index)` to the Stack. In the Structural Seed comment, specify an explicit CPU index with `tool.uv.sources` for **both** torch and torchvision. Optionally list notable transitive dependencies brought in by mantis-tsfm: `datasets` 5.1.0, plus `dill`, `multiprocess` and `xxhash`. Importing `mantis.architecture` alone does **not** import `datasets` (verified: `'datasets' in sys.modules == False`).

## F6 — Medium: unpinned weights for Mantis and timm ImageNet (conflicts with AD-10/AD-13)

**Evidence**

- AD-10's cache key and AD-13's provenance include the "weights revision". Only LWM and WavesFM have pinned shas.
- `paris-noah/Mantis-8M` is at sha `bc7d5ab40c02133386a28e2c127f35c17c86901d` and was last modified **2026-08-21**, so the repo is still changing.
- The spine says ImageNet weights are "auto-downloaded" by timm, which fetches from `timm/<model>` on HF at `main`.
- Verified `Mantis8M.from_pretrained('paris-noah/Mantis-8M', revision='bc7d5ab4…')` works: the HF `config.json` carries `"device": "cpu"`, the model reports `device cpu`, and `(3,1,512) → (3,256)` on CPU. The class default is `device='cuda'`, so CPU depends on the repo config.

**Suggested fix:** pin Mantis-8M at a full sha in the Stack and pass `device='cpu'` explicitly. For timm, choose the exact model name (e.g. `vit_base_patch16_224.augreg_in21k_ft_in1k`), then either download it with `hf_hub_download(..., revision=<sha>)` and pass `pretrained_cfg_overlay=dict(file=path)`, or record the sha timm resolved. Add a Stack row for each.

## F7 — Low: uv 0.11.14 is the local install, not current

**Evidence.** PyPI `uv` latest is **0.12.23** (2026-10-03). The 0.12 line started 2026-07-28, and 0.11.14 was released 2026-05-12. `uv --version` on this machine prints `uv 0.11.14`.

The memlog line "Verified on PyPI 2026-10-05: … uv 0.11.14" is therefore inaccurate. Everything else on that line matched PyPI latest.

**Suggested fix:** either upgrade (`uv self update`) and pin 0.12.x, or label the row "0.11.14 (installed; 0.12.23 current)" and set `[tool.uv] required-version = ">=0.11.14"`. Both versions resolve this stack. The lock above used 0.11.14.

## F8 — Low: short shas for HF pins

**Evidence.** `hf_hub_download('wi-lab/lwm-v1.1','config.json',revision='1b68512')` works and resolves to the snapshot `1b68512dbe97de56841301206eacb780fcd76011`. Full shas:
- `wi-lab/lwm-v1.1` = `1b68512dbe97de56841301206eacb780fcd76011` (lastModified 2025-04-09)
- `ahmedaboulfo/wavesfm` = `6a63011e7cbc4ea05ec664d02643e7330c9cc503` (lastModified 2026-02-16, Apache-2.0)

Both are also the current `main` heads.

**Suggested fix:** record full 40-char shas in the Stack, adapter params and cache keys (AD-10). A short sha in the hash input is a second spelling of the same revision.

## F9 — Low: import-linter claim is realistic, with one gap

**Evidence.** `import-linter` 2.15 (2026-09-04, Python ≥3.10). In the scratch project:
- A `layers` contract (`runner` > `config | adapters | results_api` > `core`) was KEPT.
- An `independence` contract with the wildcard `csi_bench.adapters.extractors.*` correctly flagged `wavesfm.adapter -> lwm._vendor`.
- `lint-imports` exits non-zero on broken contracts, so it can run as a pytest test or a CI step.

**Suggested fix:**
- Replace "import-linter style test" with "import-linter contracts in `pyproject.toml` (`layers`, `independence` over `adapters.*.*`, `forbidden` for `_vendor` outside its adapter), run by a pytest wrapper".
- Notebooks aren't part of the package, so "notebooks import only results_api" needs its own check (e.g. a test that scans `notebooks/*.ipynb` for `csi_bench.` imports) or a softer wording.

## F10 — Info: confirmed OK

- **Versions = PyPI latest on 2026-10-06:** torch 2.14.1 (2026-09-30), numpy 2.5.3, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 25.0.1, pydantic 2.13.5, PyYAML 6.0.3, huggingface-hub 2.1.1, timm 1.0.30 (2026-09-22), mantis-tsfm 1.1.0, ruff 0.16.10, ty 0.0.84 (classifier "Development Status :: 4 - Beta"), pytest 9.1.1.
- **Python 3.12:** every `requires_python` admits 3.12. numpy 2.5 requires ≥3.12, so 3.12 is the floor. The local interpreter is 3.12.3. `uv sync` installed all cp312 wheels.
- **huggingface-hub 2.x:** the breaking changes are the httpx2 transport and removal of the `huggingface-cli` binary, plus deprecated upload APIs (release notes v2.0.0). `hf_hub_download` and `PyTorchModelHubMixin` work. The WavesFM README still uses `huggingface-cli`, so use `hf` or the Python API instead.
- **timm 1.0.30:** `timm.models._manipulate.checkpoint_seq` exists. `timm.models.vision_transformer.Block` is defined, and `PatchEmbed` is re-exported there. The constructor positional order `(dim, num_heads, mlp_ratio, qkv_bias)` is unchanged, and the WavesFM block keys load against it. Upstream pinned `timm==1.0.11`, so the exact pin of 1.0.30 is justified.
- **torch.load safety:** both `.pth` files load under the torch 2.14 default `weights_only=True`. Upstream's `weights_only=False` isn't needed.
- **Mantis on CPU:** works (see F6).
- **ty/ruff config in pyproject:** `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ty.environment] python-version`, and `[tool.ty.src] exclude` all took effect in the scratch project.
- **Licenses:** LWM HF card declares none, WavesFM GitHub has none, and the HF weights are Apache-2.0. This matches the memlog's open question.

## Suggested Stack table after fixes

| Name | Version |
| --- | --- |
| Python | 3.12 |
| uv | 0.12.23 (or 0.11.14 with `required-version`) |
| torch / torchvision (explicit CPU index via `tool.uv.sources`) | 2.14.1+cpu / 0.29.1+cpu |
| pandas | 3.0.6 (+ uv `override-dependencies`, see F1) |
| mantis-tsfm + Mantis-8M weights | 1.1.0 + HF `paris-noah/Mantis-8M` @ `bc7d5ab40c02133386a28e2c127f35c17c86901d` |
| LWM v1.1 | HF `wi-lab/lwm-v1.1` @ `1b68512dbe97de56841301206eacb780fcd76011` (model + `patch_maker` excerpt only) |
| WavesFM | weights HF `ahmedaboulfo/wavesfm` @ `6a63011e7cbc4ea05ec664d02643e7330c9cc503`; code GitHub `AhmedTarek62/wavesfm` @ `0279d3b11afd7691f2565a8361a5d0d60b3ae363` |
| timm + ImageNet weights | 1.0.30 + named model @ pinned sha |
| import-linter | 2.15 |
| (others unchanged) | |
