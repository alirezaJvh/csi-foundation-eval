# Vendored LWM v1.1 excerpt

- Repository: https://huggingface.co/wi-lab/lwm-v1.1
- Revision (full sha): `1b68512dbe97de56841301206eacb780fcd76011`
- Weights: `models/model.pth` at the same revision, fetched at run time by the adapter with
  `huggingface_hub` into `.cache/weights/` (not vendored).

## Files

| File | Upstream source | Local edits |
|------|-----------------|-------------|
| `lwm_model.py` | `lwm_model.py` | none (byte-for-byte copy, sha256 `472373a08c22c107a6135e5637fd0f92310789af5faa4e6d586e31f32de0aae5`) |
| `patch_maker.py` | `input_preprocess.py` lines 259-301 (`patch_maker`) | excerpt only: kept the `patch_maker` function body unchanged and prepended `import numpy as np`; every other function and import (DeepMIMO, tqdm, utils, pickle, torch, ...) was dropped. Upstream file sha256 `00ab758af1f8687dea95bc43935a44a25b130abca29ffdcd27b9c47dd3deb421` |
| `__init__.py` | — | new, empty |
| `SOURCE.md` | — | new, this file |

Nothing here imports DeepMIMO. The adapter never calls `lwm.from_pretrained`, `tokenizer`,
`make_sample` or `patch_gen` (`patch_gen` calls `patch_maker` without its row/col arguments,
an upstream bug).

## Constructor arguments

The adapter builds the model with the `lwm()` defaults:

| Argument | Value |
|----------|-------|
| `element_length` | 32 |
| `d_model` | 128 |
| `n_layers` | 12 |
| `max_len` | 513 |
| `n_heads` | 8 |
| `dropout` | 0.1 (inactive: the model is kept in `eval()`) |

The CLS token is `0.2 * np.ones(32)`, the constant from upstream `tokenizer`, re-implemented in
the adapter.
