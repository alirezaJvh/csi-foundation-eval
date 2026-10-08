## Deferred from: code review of spec-1-2-load-the-own-recordings-as-labelled-recordings (2026-10-07)

- `TorVergata.Params.root` default (`data/csi_proc_python/saved_csi`) is relative to CWD; running outside the repo root fails with "dataset root not found". Spec mandates the default; revisit path resolution with the CLI/config stories.
