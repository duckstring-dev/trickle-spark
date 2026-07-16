# Plan: trickle-spark — the DBSP incremental engine on Spark + Delta Lake

> The design document the port was built against, carried over verbatim from the duckstring repo
> (`plans/trickle-spark.md`) when the package was extracted here — "this repo" in the text below
> means duckstring, where the port was built in-tree beside the reference implementation.

Status: **built through Phase 4** (core io, builder DAG, aggregate, accumulate — 41 tests green on a
local Delta session; Phase 5 docs/CI landed with it). Deferred within the build, beyond the plan's own
deferred list: two-variable co-moments + `agg.reduce`, ungrouped scans, in-place output schema
evolution. A standalone port of Trickle's DBSP semantics (Z-sets, changed-key
incremental joins, retractable aggregation) to **Apache Spark + Delta Lake**, aimed first at Databricks
but engine-wise pure OSS. Built in this repo for proximity to the reference implementation and its test
suite, then **extracted to its own repo/distribution** (`trickle-spark` on PyPI) once it stands — the same
pattern as `playground/`. It is a *sibling implementation*, not a shared-core refactor: `duckstring/trickle/`
stays untouched and serves as the **reference implementation and behavioural oracle**.

## Goals (what the port must deliver)

1. **Update each table according to changes only** — DBSP Z-set semantics with the changed-key
   (affected-key) recompute, so general **tree/bushy joins** with updates and deletes flowing through them
   are maintained in O(change), not O(state).
2. **Periodic push-style scheduling** — batch jobs on a timer (or a table-update trigger); no resident
   service, no streaming runtime.
3. **Heterogeneous cadences** — different subgraphs run at different frequencies with **zero coordination**
   between them; a slow consumer of a fast producer just reads a wider change window.

## Non-goals (deliberate deviations from Duckstring)

- **No epochs / freshness.** There is no `_duckstring_f`, no user-supplied freshness axis, no explicit
  changelog table. The change axis is the **Delta table version**; windows are version ranges. (Duckstring's
  epoch model exists to make windows content-addressed and replay-stable; here replay safety falls out of
  the watermark scheme — see *Exactly-once* below. An explicit-changelog change source can be added later
  behind the change-source seam if a cross-system use case demands it.)
- **No orchestration.** No pond/Catchment/Duck analog, no demand model, no triggers. Scheduling is the
  platform's job (Databricks Workflows / any scheduler). The package is a **library**: one idempotent
  function call per output table.
- **No hand-rolled storage tiering.** No 3-tier LSM, no `fold_warm`/`checkpoint`, no reconstruct-on-read,
  no parquet parts/sidecars. Delta *is* the log-structured table; `OPTIMIZE`/`VACUUM` (or Databricks
  predictive optimization) is the compaction story. The output **main is a real, materialised Delta table**.
- **Nothing Databricks-specific in the engine.** Everything load-bearing — CDF, time travel, `MERGE`,
  commit metadata — is OSS Delta (`delta-spark`), so the engine runs and tests on a local Spark session.
  Databricks (Workflows, Asset Bundles, Unity Catalog, serverless, predictive optimization) is the
  *deployment layer*. A `trickle_spark.databricks` module is reserved for genuinely DBR-only conveniences;
  v1 ships it empty or not at all.

## Core model

Every node in the user's graph is a **Delta table with Change Data Feed enabled** (the package sets
`delta.enableChangeDataFeed=true` whenever it creates or first writes a table). CDF is the changelog:
a `MERGE` emits row-level `insert` / `delete` / `update_preimage` / `update_postimage` rows per commit,
which map exactly to a **full-row Z-set** — pre-image → weight −1, post-image/insert → +1, delete → −1.
Because deltas carry full row images (not key tombstones), the builder can join on **any** key and
propagate deletes soundly — the same property the Duckstring changelog format was designed for, obtained
here for free from the storage layer.

One run of one output table:

1. **Plan.** Read the output's watermarks (see below). **Pin** each source's current version. The window
   per source is `(last_processed_version, pinned_version]`.
2. **Read deltas.** `table_changes(source, lo+1, hi)` per source, mapped to a Z-set and **consolidated**
   (full-row `GROUP BY`, `SUM(d) <> 0`) so multi-commit windows collapse (a→b→c→d = one net change).
   An empty window makes that source a free stable operand — its join terms drop out.
3. **Compute ΔO.** Per join node, the **affected-key recompute** (identical to `plans/trickle-dag.md`):
   `K = πₖ(δL) ∪ πₖ(δR)`; recompute the join restricted to `key ∈ K` over the **new** states (+1) and the
   **old** states (−1); consolidate. All six `how`s, bushy shapes, one rule.
   - **New state** = `source VERSION AS OF pinned`. **Old state** = `source VERSION AS OF last_processed` —
     **time travel replaces reconstruction** (`reconstruct_current`/`_reconstruct_old` have no analog here;
     the Delta transaction log is the state history).
   - The `key ∈ K` restriction is a semi-join pre-filter on **both** inputs — broadcast the (small) key set
     so a small change never scans the big side. This is where Spark/Photon genuinely pays off.
4. **Apply.** Consolidate ΔO by the output PK and apply with **one `MERGE INTO`** (net-negative keys →
   `DELETE`, present rows → update/insert) — the delete-then-insert Z-set apply, as one atomic commit.
   That commit's CDF is the next consumer's input; the graph composes.
5. **Advance watermarks in the same commit** (below).

All-windows-empty → **skip**: no commit at all, nothing downstream fires. (The no-change-skip analog,
structural rather than flagged.)

### Watermarks live in the output's own commit

`(output, source) → last processed source version`, stored as **custom commit metadata**
(`userMetadata`, a JSON map `{source_fqn: {version, table_id}}`) on the output's write commit — the
"watermark lives in the destination" pattern proven by `egress/postgres.py`, which makes the data advance
and the watermark advance **one atomic Delta commit**. Recovery reads the output's history for the latest
trickle-stamped commit. Details that make it robust:

- **Pin at plan time.** Read CDF and states *as of* the pinned versions and record the pinned versions;
  concurrent producer commits mid-run land cleanly in the next window.
- **Record the source's Delta `table_id`** beside its version. A dropped-and-recreated source resets its
  version counter; an id mismatch means the watermark is meaningless → comprehensive fallback, never a
  silently wrong window.
- **Comprehensive runs advance watermarks too** (they read sources at pinned versions like any run).
- **No control table.** If ops wants to *see* watermarks, expose a helper that derives a view from table
  history — never a second source of truth.

### Exactly-once without epochs

- Crash **mid-run**: nothing committed; the re-run re-plans from unchanged watermarks. Clean.
- Crash **after** the commit: watermarks advanced atomically with the data; the re-run sees empty windows
  and skips. Replay safety falls out of the scheme — no idempotent-writer bookkeeping, no run keys.
- **Single writer per output table** is the standing assumption (as it is for a Duckstring major line);
  Delta's optimistic concurrency is the backstop against an accidental second writer, not the mechanism.

### Fallback rungs (the comprehensive path)

A source reads as **full** when: no watermark exists (bootstrap), the window is unreadable (watermark older
than CDF retention / history aged past `delta.logRetentionDuration` — the *floor*/coverage-miss analog,
now a storage-retention knob), the `table_id` changed, or the consolidated delta touches more than the
per-source change fraction **`p`** (same threshold semantics as `pond.trickle(ref, p=…)`; `p=1.0` disables).
Fullness propagates up the DAG; a full subtree recomputes wholesale. Then:

- **Bootstrap** (output doesn't exist): full compute → create the table (CDF on) with watermark metadata.
- **Fallback with an existing output**: full compute, then **diff against the current output and `MERGE`
  only the real changes** — the `merge_table`-diff analog. This matters: a blind overwrite would emit a
  whole-table CDF and cascade comprehensive recomputes downstream; the diff-then-MERGE keeps downstream
  windows honest and small.

Retention expiring is a *lag SLA, not correctness* — exactly the Duckstring retention stance, with
`delta.deletedFileRetentionDuration`/`delta.logRetentionDuration`/CDF retention as the knobs.

## The library contract — the package owns all side state

A pure library call, no control plane, no job-start process:

```python
import trickle_spark as ts

ts.run(spark, output="cat.schema.priced", plan=...)
# discovers its own watermarks → plans windows → computes ΔO → one MERGE commit → done
```

Everything the engine needs lives in tables it already writes: watermarks in output commit metadata;
aggregation/accumulator state in **companion Delta tables** in the same schema, created lazily on first
use; CDF enablement set at table creation. Schema creation, grants, and maintenance belong to the
deployment layer (Asset Bundle / Terraform / predictive optimization) — the moment a setup pass looks
necessary, something is in the wrong layer.

**Companion-state atomicity** (the one real seam in the no-control-plane story): a state companion and its
output are two tables and can't commit together. Order: **state first, output second**, each commit
carrying the same pinned-version map. A re-run after a crash-between compares the two watermark maps,
sees the state already at the pinned versions, and **fast-forwards** to the output write (the state update
is idempotent because it's version-keyed). This replaces the epoch-keyed (`f`-stamped) replay guards of
`apply_aggregate`/`apply_accumulate`.

## The "pond" question — a deployment pattern, not an engine concept

The engine's unit is strictly **one output table** with per-source watermarks. Nothing is ever
co-scheduled for correctness. The grouping survives as a documented **deployment pattern** (README, not
code): **repo + Databricks Asset Bundle + one UC schema + one Workflows job** whose tasks run the schema's
outputs in dependency order. Edges *inside* a bundle are task `depends_on`; edges *across* bundles are
watermark reads with zero coordination; bundles run at whatever cadence they like (timer or table-update
trigger). Session sharing within a task (shared scans / cached intermediates across sibling outputs) and
single-writer discipline (the bundle's principal owns its schema's writes) are the practical payoffs.
Catalog-per-environment layers on cleanly. The engine must never *require* any of this.

## Surface (what ports, what changes)

- **Builder DSL** — ports structurally intact from `trickle/builder.py`: the operator DAG of binary
  incremental joins (all six `how`s, bushy/snowflake), `.filter`/`.mutate`/`.select`/`.alias`, the
  key pre-filter, per-node temp views composed to inline SQL, `p`-threshold, `ivm=False`/`key_filter=False`
  escapes. The **source handle changes**: `ts.source("cat.schema.orders")` (a table reference) replaces
  `pond.trickle(...)`; the terminal is `.merge_into("cat.schema.out", pk=…)` (and later `.append_to(...)`).
  Dialect shims: `"quoted"` → `` `backticks` ``, `* EXCLUDE` → `* EXCEPT`, `GROUP BY ALL` (native in
  Spark 3.4+), catalog introspection via the Spark catalog API. **Preserve the `unique_name` per-run
  view-naming discipline** — Spark temp views have the identical shared-name rebinding gotcha.
- **`.sql()` escape hatch** — same contract (collapse to a materialised relation, comprehensive from there,
  output delta stays incremental via the diff). Spark SQL string or an ibis expr compiled with
  `dialect="databricks"`/`"pyspark"` (ibis stays a lazy non-dependency).
- **`agg`** — the accumulator algebra ports near-verbatim (the folds and Chan/Pébay centred-moment merges
  are SQL expressions); state in a `{output}__aggstate` companion; the retraction-rescan families
  (`RESCAN_KINDS`) keep the same treatment. Group-count-to-zero drops + retracts, as today.
- **`acc`** — the order-dependent scans keep the carried-state companion design, but the Python fold's
  execution vehicle becomes **`applyInPandas` per group** (the one module that's a rewrite rather than a
  translation). The SQL-window fast path for the windowable metrics ports directly.
- **`Delta` (the read surface)** — same shape (`.zset`/`.is_full`/`.upserts`/`.deletes`) as a DataFrame-
  carrying object, for user code that consumes raw deltas; backed by `ts.delta_of(table, since=…)`.
- **The change-source seam** — `delta_of(table, watermark) -> Z-set window` is a small interface with the
  CDF implementation behind it, so an explicit-changelog backend (epochs, cross-system draws) can be added
  later without touching the join engine.
- **System namespace** — decided day one, can't migrate later: the package owns **`_trickle_`**
  (`_trickle_d` on in-flight Z-sets, `_trickle_` state-companion columns, the commit-metadata key).
  Persisted **outputs are clean tables** — with CDF as the changelog and a materialised main, system
  columns exist only in-flight and in state companions. Reject user columns under the prefix at write.
- **Determinism contract, extended for Spark**: retractions cancel by full-row identity, so projections
  must be deterministic (no `now()`/`rand()`), **no map-typed columns** in Z-set rows (Spark can't
  `GROUP BY` maps), and float-producing expressions deserve a warning (recompute jitter surfaces as
  phantom retract/insert pairs).

Deferred from v1 (in rough priority order): the `.append_to` terminal with the droplog/conflict semantics
(plain Delta appends cover the easy cases; the conflict machinery earns its keep later), the explicit-
changelog change source, holistic aggregates / DISTINCT (stay a downstream `.sql()` step, as in the
reference), skewness/product, cross-run persisted intermediate traces, streaming ingestion (batch-at-a-
window only — the model that keeps all of this tractable).

## Repo layout, testing, extraction

- **`trickle-spark/` at the repo top level**, fully self-contained: own `pyproject.toml` (deps: `pyspark`,
  `delta-spark`; module `trickle_spark`), own `tests/`, own README (which is where the deployment pattern
  is documented). **It must import nothing from `duckstring`** — extraction is `git mv` to a new repo.
  Not part of the duckstring wheel; excluded from its build like `playground/`.
- **Tests run on a local Spark session** with delta-spark configured (session-scoped fixture), fully
  offline. They are slow relative to the DuckDB suite: mark them (`-m spark`), exempt them from the 5 s
  `pytest-timeout` default, and give them their own CI job so the duckstring suite stays fast.
- **The oracle**: port the behavioural scenarios from `tests/test_trickle.py` / the builder+agg suites and
  assert the same observable behaviour — replay/idempotence (re-run → empty windows → skip), retraction
  propagation through tree joins, outer-join incomparable maintenance, `p`-threshold and retention-expiry
  fallbacks (diff-then-MERGE keeps downstream windows small — assert on the *downstream* CDF), agg
  retraction rescans, the companion-state crash-between fast-forward. Where the reference asserts on
  epoch mechanics, translate to version-window mechanics; where it asserts on Z-set outcomes, port verbatim.
- A Databricks-workspace smoke test (UC three-part names, Workflows wiring) is manual/live for v1 — the
  engine's CI never needs a workspace.

## Build order

1. **Core io** (`io.py` analog): CDF→Z-set mapping + consolidation, the change-source seam, watermark
   commit-metadata read/write with pinning + `table_id` guard, the MERGE apply, bootstrap + comprehensive
   diff-then-MERGE, `p`-threshold, `ts.run` skeleton. This alone is useful (hand-written delta logic over
   managed watermarks).
2. **Builder DAG**: sources/joins (all `how`s) + the pipeline ops + `*`-output rules, ported from
   `trickle/builder.py` with the dialect shims; `.merge_into` terminal; `.sql()`.
3. **`agg`**: state companions + the incremental/rescan/comprehensive paths + the state-first commit order.
4. **`acc`**: `applyInPandas` folds + carried state + the windowed fast path.
5. **Deployment pattern docs + a worked example bundle** (two cadence tiers, a cross-bundle edge), and the
   extraction checklist.

Each phase lands with its slice of the ported oracle suite green on local Spark.
