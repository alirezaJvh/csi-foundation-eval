# Sprint Change Proposal: share `paper_beamforming` per-position metrics across the sweep (Story 2.11)

Date: 2026-10-10 · Author: Jj (with Claude) · Mode: Batch · Scope: Moderate
Input: `_bmad-output/implementation-artifacts/story-2-11-sweep-speedup-explained.html` (idea A), `proposal-2-10-reuse-inner-windows.md`

## 1. Issue summary

Story 2.10 made `paper_beamforming` compute each inner-window metric once per packet within one run (extraction 956 s → 365 s on the baseline's first run; matrix bit-identical). The sweep still repeats that work: the six per-position arrays do not depend on `window_len`, `stride` or the room set, yet `experiments/classical_baseline.yaml` recomputes them in 12 runs (4 window lengths × 3 room sets). r1+r2 is the union of the r1 and r2 blocks. Estimated extraction for the sweep: about 1.6 h today, about 15 min with sharing (estimates; r2 assumed about the size of r1).

## 2. Impact analysis

- **Epics:** Epic 2 gets one new story (2.11). Epics 3–5 unchanged; Epic 5's full sweep benefits.
- **PRD:** no conflict.
- **Architecture:** AD-10 says the cache holds "one extractor's matrix over all windows". A second, runner-owned entry kind (per-block intermediates) needs one added line. Still only the runner reads and writes it; extractors never touch the cache; deleting `.cache` never changes a result. AD-7/AD-8 (the `extract_blocks` path) are unchanged.
- **Code:** `core/ports.py` (optional context methods), `adapters/extractors/paper_beamforming.py` (split compute from slice), `runner/run.py` (get or compute and store per-block arrays), `adapters/storage` (store for arrays, or reuse the npz cache), tests.
- **Leakage:** safe. Arrays are per recording block, so nothing crosses a split gap; each window reads only positions inside its own packets.
- **Out of scope:** several sweep processes at once (idea B, needs run and cache-key locks), and the extractor math rewrite.

## 3. Recommended approach

**Direct Adjustment:** add Story 2.11 after 2.10. Design:
- The extractor may offer `compute_context(block) -> arrays` and `extract_blocks(block, offsets, window_len, context=...)`. Per-position arrays cover the whole block, not only the offsets of one run, so every window length and stride can slice them.
- The runner keys each entry per recording block: dataset, `recording_id`, block range, preprocessing chain with fitted-state hash, extractor name, params affecting results (`fix_distance_loop`), `VERSION`. It does not include `window_len`, `stride` or the room set, so r1+r2 reuses the r1 and r2 entries. Preprocessing with global fitted state yields different keys and so no wrong sharing.
- The feature-matrix cache key, matrix and `VERSION` stay as they are, so run ids do not change.

Rejected: the sweep-wide in-memory cache (lost on restart), and merging this with parallel sweep processes (needs locks; a separate story if still wanted). Effort moderate, risk low because of exact-equality tests. Main risk: disk use of the arrays (6 float64 per packet, small) and a correct key.

## 4. Detailed change proposals

1. **`planning-artifacts/epics.md`:** append Story 2.11 after 2.10, and change 2.10's "Out of scope" cache sentence to point to 2.11. Story 2.11 acceptance criteria:
   - Given two runs of the same extractor with different `window_len` or room sets, when the second runs, then it reads the per-block arrays from the cache and recomputes none that exist.
   - Given any window length, stride or room set, then the feature matrix equals the one from Story 2.10's path exactly (including against the previously cached old matrix), for `fix_distance_loop` true and false.
   - Given r1+r2, then it is built from the r1 and r2 entries.
   - Given a preprocessing chain with fitted state, then the entries are keyed by that state, so values are never shared across different state.
   - Given `.cache` deleted, then results are unchanged. Feature-matrix cache key, `VERSION` and run ids are unchanged.
   - Given the baseline sweep's paper_beamforming runs, then total extraction time before and after and peak RSS (about 4 GB or less) are recorded in the story spec.
2. **`ARCHITECTURE-SPINE.md` AD-10:** add: "The runner may also store per-block intermediates an extractor with `uses_context = True` offers (keyed per recording block, without `window_len`, `stride` or room set). Same rules: one client, complete keys, atomic writes, deletion never changes a result."
3. **`sprint-status.yaml`:** add `2-11-share-paper-beamforming-per-position-metrics-across-the-sweep: backlog`. **`epic-2-context.md`:** add Story 2.11 to the list and one sentence under Cache.
4. **`deferred-work.md`:** log "parallel sweep processes (idea B)" and "extractor math rewrite (per-packet amplitude and pair distances)" as follow-ups.

## 5. Implementation handoff

- **Scope:** Moderate: backlog and architecture amendment, then Developer agent.
- **Order:** finish the Story 2.10 code review, then apply edits 1–4, then `bmad-build 2.11`.
- **Success criteria:** identical feature matrices, cache key and run ids unchanged, sweep extraction time measured before and after, peak RSS within about 4 GB.
