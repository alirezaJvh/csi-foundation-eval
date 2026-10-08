## Deferred from: code review of spec-1-2-load-the-own-recordings-as-labelled-recordings (2026-10-07)

- `TorVergata.Params.root` default (`data/csi_proc_python/saved_csi`) is relative to CWD; running outside the repo root fails with "dataset root not found". Spec mandates the default; revisit path resolution with the CLI/config stories.

## Deferred from: code review of spec-1-4-inspect-the-dataset-with-csi-bench-describe (2026-10-08)

- Repo-wide `ruff check` / `ruff format --check` fail (181 errors, all under untracked `.agents/skills/**`), so story 1.4's AC3 verification command does not pass in this workspace. Environmental/pre-existing, not caused by the change — fix by excluding the BMad skills tree from the ruff config.
