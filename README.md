# trickle-spark

DBSP-style incremental data transforms on **Spark + Delta Lake**: update each table according to
changes only, with Z-set (weighted-row) semantics and changed-key join maintenance, so updates and
deletes propagate correctly through general tree joins — at the cost of the change, not the size of
the state.

trickle-spark is a standalone sibling of [Duckstring](https://github.com/duckstring-dev/duckstring)'s
Trickle engine (the reference implementation), re-based on what the lakehouse already provides
(the full design record is [docs/design.md](docs/design.md)):

- **Delta Change Data Feed is the changelog.** A `MERGE` into a Delta table emits row-level pre/post
  images — exactly a full-row Z-set. Nothing is maintained by hand; every CDF-enabled table
  participates.
- **Time travel is the old state.** The prior state of a source is `VERSION AS OF` its watermark —
  nothing is reconstructed.
- **Watermarks live in the output's own commit.** Each run lands **one commit** carrying both the data
  change and the map `{source → last processed version}` as commit metadata — exactly-once by
  construction, with no control table and no second source of truth.
- **No epochs, no control plane, no resident service.** A run is a pure library call. Scheduling
  belongs to your scheduler; different outputs can run at different cadences with zero coordination,
  because every consumer just reads its own window of each source's change feed.

The engine is pure OSS Spark + Delta (`delta-spark`) — it runs and tests on a bare local session.
Databricks (Workflows, Asset Bundles, Unity Catalog, predictive optimization) is the *deployment
layer*, not a dependency.

## Install

```
pip install trickle-spark        # (from source while unreleased: pip install -e .)
```

## Quickstart

```python
import trickle_spark as ts

# orders ⋈ catalog, maintained incrementally: a catalog price change recomputes
# only the affected order rows, and deletes propagate.
plan = (
    ts.source("shop.orders").alias("o")
    .join(ts.source("shop.catalog").alias("c"), on="product_id", how="left")
    .mutate(amount="o.qty * c.price")
    .select("o.order_id, o.product_id, o.qty, c.price, amount")
)
plan.merge_into(spark, "shop.priced", pk=("order_id",))
```

Run that on any schedule. Each invocation discovers its own watermarks from `shop.priced`'s commit
history, pins the sources, reads only the CDF windows since the last run, computes the output's
Z-set delta, and applies it with one `MERGE` — whose own change feed is the next consumer's input, so
plans compose into pipelines across tables, jobs, and cadences.

The first run **bootstraps** (creates the table, CDF enabled). A run with nothing to do **skips**
without committing. A window that's unreadable (retention expiry, a recreated source) or too large
(the per-source change fraction `p`, default 0.3) falls back to a **comprehensive** recompute that is
*diffed against the current output* — so even a full recompute emits only real changes downstream.

### Aggregation

```python
from trickle_spark import agg

(ts.source("shop.priced")
 .aggregate(by="product_id",
            orders=agg.count(), revenue=agg.sum("amount"),
            avg_price=agg.mean("price"), sd=agg.stddev("price"))
 .merge_into(spark, "shop.revenue_by_product"))   # pk defaults to the group key
```

Distributive metrics fold in O(δ) from accumulators kept in a `…__trickle_aggstate` companion table;
`var`/`stddev` are maintained by the numerically stable Chan/Pébay centred-moment merge; `min`/`max`/
`argmin`/`argmax`/`bool_*`/`bit_*` extend in place on inserts and rescan a group's current membership
only when it saw a retraction. Also available: the weighted family (`weight_total`, `weighted_sum`,
`weighted_average`), a retractable `product`, and the two-variable co-moments (`covariance`,
`pearson_correlation`, `ols_slope`, `ols_intercept`) maintained by the same Pébay merge over pairwise
non-NULL rows. `agg.reduce(fn, init)` is the order-dependent exception — a custom fold in `.along`
order collapsed to one value per group (a tail append resumes the group's carried state; a past edit
re-folds it).

### Ordered scans

```python
from trickle_spark import acc

(ts.source("market.ticks")
 .along("event_time")
 .accumulate(by="symbol",
             run_total=acc.sum("qty"), smooth=acc.ema("price", 0.3),
             prior=acc.prev("price"))
 .merge_into(spark, "market.ticks_scored", pk=("tick_id",)))
```

`.accumulate()` enriches **every row** with running values in `.along` order within its group. A
tail-only append resumes each group's carried fold-state (O(new)); an edit or delete in a group's past
re-folds that group and diffs, so only rows whose running values actually changed hit the output's
change feed. `acc.scan(fn, init)` is the custom-fold escape hatch. Omit `by` for an **ungrouped**
scan over the whole table — sound, but one serial fold by construction.

### Append-only outputs

```python
(ts.source("shop.orders").alias("o")
 .filter("o.qty > 0")
 .append_to(spark, "shop.order_log", pk=("order_id",)))
```

For a **monotonic** transform — output rows only ever added, never updated or retracted —
`.append_to` lands pure insert commits (the output's CDF carries no updates or deletes) instead of a
MERGE. A retraction reaching the output, or a `pk` colliding with a *different* image, is a
**conflict**: the default raises before writing anything; `fail_on_conflict=False` drops the
conflicting rows (history wins, the past stays frozen) and, with `log_drops`, records them in a
`{output}__trickle_droplog` companion table. An identical image is a benign idempotent skip, so a
comprehensive re-derivation never spuriously fails — and never rewrites the past.

### Escape hatch

```python
(plan.alias("t")
 .sql("SELECT product_id, percentile(price, 0.5) AS p50 FROM t GROUP BY product_id")
 .merge_into(spark, "shop.p50", pk=("product_id",)))
```

`.sql()` breaks incremental *compute* but keeps incremental *output* — the result is still diffed, so
downstream windows stay small. Use it for anything the incremental op set doesn't cover.

### The low-level API

The builder is sugar over `ts.run`, which is useful on its own for hand-written delta logic:

```python
def full(ctx):   # the complete output from pinned reads
    return ctx.new("shop.orders").where("qty > 0")

def delta(ctx):  # the output's Z-set change from the windows (or None → comprehensive)
    d = ctx.delta("shop.orders")
    return None if d.is_full else d.zset.where("qty > 0")

ts.run(spark, "shop.clean_orders", sources=["shop.orders"], pk=("order_id",), full=full, delta=delta)
```

`ctx.delta(src)` is a consolidated Z-set (`_trickle_d` = +1/−1, full row images); `ctx.new`/`ctx.old`
are the pinned current and watermark states.

### Bridging to another incremental engine

The Z-set is the lingua franca, so crossing into or out of a sibling engine (say, a
[Duckstring](https://github.com/duckstring-dev/duckstring) Trickle pipeline where one stage needs
multinode compute) takes two primitives, both in `trickle_spark.bridge`:

```python
# inbound: land a foreign Z-set window on a Delta table — CDF re-derives the changelog from there,
# so downstream plans are fully native. `tag` (the producer's epoch) makes replays exactly-once.
ts.apply_changes(spark, "bridge.orders", zset, pk=("order_id",), tag=str(f))

# outbound: the output's consolidated change since a pin the *consumer* remembers, plus the next pin
delta, pin = ts.table_changes(spark, "bridge.priced", last_pin)
```

Neither engine learns the other's clock — each windows by its own axis, and the crossing carries full
row images. A worked Duckstring↔Spark round trip (landing ripples, epoch-keyed pin bookkeeping,
replay idempotence on both sides) is `tests/test_duckstring_bridge.py`.

## The deployment pattern ("a pond, demoted to a pattern")

The engine's unit is strictly **one output table**; nothing is ever co-scheduled for correctness.
What remains of pipeline structure is a deployment convention, and on Databricks it maps cleanly:

- **Repo + Asset Bundle + one UC schema + one Workflows job** per group of related outputs. The job's
  tasks run the schema's outputs in dependency order (`depends_on`); each task is just
  `plan.merge_into(...)` calls sharing a Spark session (shared scans and caches).
- **Edges across bundles are watermark reads** — zero coordination. A slow consumer of a fast producer
  reads a wider window; a fast consumer of a slow producer skips.
- **Cadence tiers**: one job per cadence (5-minute hot paths, hourly warm, nightly heavy). Timer
  schedules or table-update triggers both work — the runs are idempotent either way.
- **Single writer per output table.** The standing assumption everywhere. A bundle owning its schema's
  writes (via its service principal) makes it an organizational fact; Delta's optimistic concurrency
  is the backstop, not the mechanism.

Nothing in the package requires any of this — schema creation and grants belong to the bundle,
table maintenance to `OPTIMIZE`/`VACUUM` (or Databricks predictive optimization), and everything
run-scoped happens lazily inside the call.

## Semantics, fine print

- **Determinism contract.** Retractions cancel by full-row identity: projections must be deterministic
  (no `now()`/`rand()`), no map-typed columns in Z-set rows (Spark can't group by them), and
  float-producing expressions deserve care (recompute jitter surfaces as phantom retract/insert
  pairs).
- **Retention is a lag SLA, not correctness.** A watermark older than CDF/log retention degrades to
  the same comprehensive fallback as a bootstrap. Tune `delta.deletedFileRetentionDuration` /
  `delta.logRetentionDuration` to your longest acceptable gap between runs.
- **Replay safety without epochs.** Data + watermarks commit atomically; a crashed run either
  committed nothing (clean re-plan) or everything (re-run skips). The two-table paths (agg/acc state
  companions) commit state first, carrying the run's pins, and fast-forward on replay.
- **Recreated sources are detected** by table id and read comprehensively — never as a silently wrong
  window.
- **Schema evolution is out of scope**: an output's shape change is a new table (or a drop and
  re-bootstrap), not an in-place evolution.

## Status / not yet ported

The reference implementation and behavioural oracle is duckstring's `trickle/` package; the port
covers the core io, the full join DAG (all six `how`s, bushy shapes, the affected-key recompute with
a broadcast null-safe key pre-filter), aggregation — including the two-variable co-moments and the
order-dependent `agg.reduce` — ordered scans (grouped and ungrouped), and the `.append_to` terminal
with droplog conflict semantics. The reserved "explicit-changelog change source" resolved into the
**bridge pair** (`apply_changes`/`table_changes`) once the cross-system use case arrived: landing a
foreign Z-set and letting CDF re-derive the changelog covers it without reconstruction-based read
machinery in the engine. Remaining follow-ups are the reference's own deferred set (skewness,
holistic aggregates / `DISTINCT`), which stay a downstream `.sql()` step here as there.

## Development

```
pip install -e .[dev]
pytest            # a local Delta session — offline, no workspace needed
ruff check .
```

Spark 3.5's bundled Arrow needs a JDK ≤ 17 for `applyInPandas`; the test fixture picks an installed
JDK 17 automatically when the ambient JDK is newer. Use Python 3.10–3.12 — pyspark 3.5's pandas seam
still imports `distutils`, which 3.13 removed, so the `applyInPandas` paths break there.

Releases are tag-driven: pushing a `v*` tag matching the package version runs the test suite, builds,
and publishes to PyPI via Trusted Publishing (`.github/workflows/release.yml`).
