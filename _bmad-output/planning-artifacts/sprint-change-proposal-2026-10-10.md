# Sprint Change Proposal: reuse inner-window metrics in `paper_beamforming` (Story 2.10)

Date: 2026-10-10 · Author: Jj (with Claude) · Mode: Incremental · Scope: Moderate
Input: `_bmad-output/implementation-artifacts/proposal-2-10-reuse-inner-windows.md`

## 1. Issue summary

Story 2.9 added multi-core extraction, but `paper_beamforming` got no faster on the first run of `experiments/classical_baseline.yaml` (window_len 20, r1, svm; 37,937 windows):

| workers | extraction | whole run | peak RSS |
|---|---|---|---|
| 1 | 956 s | 16:29 | 1.33 GB |
| 4 | 989 s | 17:03 | 1.32 GB |

The larger cause is redundant work. With stride 1, every window recomputes inner-window metrics that its neighbours already computed (about `L-16` times per metric). The "memory-bandwidth bound" diagnosis comes from microbenchmarks and is unconfirmed.

## 2. Impact analysis

- **Epics:** Epic 2 gets one new story (2.10). Epics 3–5 are unchanged in scope. Epic 5's full sweep benefits most.
- **PRD:** no conflict.
- **Architecture:** AD-7 (extractors get only their window's slice) and AD-8 ("pure per input") conflict with block-aware reuse and are reworded. AD-10 is unaffected: the cache key and the cached matrix stay the same.
- **Code:** `core/ports.py` (optional capability), `runner/run.py::_extract`, `adapters/extractors/paper_beamforming.py`, plus new tests.
- **Leakage:** safe. Every inner position a window uses lies inside the window (positions `a+1 … a+L-16`), and block-wise computation never crosses a split gap.
- **Estimated gain (from operation counts, not measured):** distance computations per window fall from `L-16` to 1 per packet, so about 4× at L=20 and about 34× at L=50.

## 3. Recommended approach

**Direct Adjustment.** Add Story 2.10 with an optional block-aware extractor method (`uses_context = True`, `extract_blocks`). The slice-based `extract` remains the reference, and a contract test asserts exact equality. Rollback and MVP review were rejected: 2.9's work is sound and scope is unchanged. Effort is moderate and risk is low because of the exact-equality tests. Sharing per-position arrays across window lengths is deferred, since it adds only about 4× more and needs a runner-owned cache layer.

Risks for the story: peak memory when all positions of a block are processed at once (chunk it, stay within ~4 GB), and bit-equality (if impossible, the tolerance decision goes to the human and `VERSION` is bumped).

## 4. Detailed change proposals (all approved)

1. **`planning-artifacts/epics.md`:** replace the "Out of scope" paragraph of Story 2.9 with a pointer to 2.10, and append Story 2.10 (ACs: stage timings first; exact equality for `fix_distance_loop` true/false and paper parity; runner calls blocks never across a split gap and a contract test covers every extractor offering both paths; cache key and VERSION unchanged when values are identical; before/after time and RSS recorded, RSS within ~4 GB).
2. **Architecture spine:**
   - AD-7 allows `extract_blocks` for extractors that declare `uses_context = True`.
   - AD-8 rewords purity (same features alone, in a batch, or in a block; sharing only via `extract_blocks`, tested).
   - AD-8 adds the `uses_context` declaration.
   - The same note goes into the `core/ports.py` docstring and `epic-2-context.md`.
3. **`sprint-status.yaml`:** add `2-10-compute-paper-beamforming-inner-metrics-once-per-packet: backlog`. **`epic-2-context.md`:** add Story 2.10 to the list and one sentence to "Extractors and pooling".
4. **`deferred-work.md`:** update the 2.9 measurement entry. Log three follow-ups: default extraction workers to 1 (decide in the 2.9 review), per-stage timing (first AC of 2.10), and confirm the memory-bandwidth diagnosis.

## 5. Implementation handoff

- **Scope:** Moderate. Backlog update plus an architecture amendment.
- **Order:** finish the Story 2.9 code review and commit it first. Then apply edits 1–4. Then create and implement Story 2.10 via the dev agent.
- **Success criteria:** the feature matrix is exactly equal to today's, the paper-parity test passes, the extraction time of the baseline's first run drops by the measured factor, and peak RSS stays within ~4 GB.
