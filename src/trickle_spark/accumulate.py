"""Order-dependent per-row scans: the ``.along(...).accumulate(by, **metrics)`` machinery.

The **merge-mode** ordered scan of the reference implementation (duckstring
``trickle/io.py:apply_accumulate_merge``): each output row is its input row enriched with running
values in ``.along`` order within its ``by`` group. Affected groups split two ways —

- **tail-only** (no retraction, every changed row strictly beyond the group's carried ``.along``
  high-water mark, or a brand-new group): the fold **resumes** from the carried per-group state and
  touches only the new rows — O(new), the common streaming case;
- **past-changed** (a retraction, or an edit at/below the high-water mark): the group is **re-folded**
  over its current membership and merge-diffed against the existing output — O(group), and only rows
  whose running values actually changed survive the diff into the output's change feed.

The fold executes as **``applyInPandas`` per group** — a Python fold, so every metric (the recursive
``ema``/``tema``, the FIFO-buffer ``lag``/``convolution``, a custom ``scan``) is handled uniformly.
Carried state is JSON per group in a ``{output}__trickle_accstate`` companion Delta table
(``by`` + ``_hw`` the along high-water + ``_state``), committed **state-first with the run's pins**
like the aggregate state (see ``aggregate.py``). One replay subtlety is unique to the scan: on a
fast-forward (state already at this run's pins) a tail resume would fold the window **twice** — so a
fast-forward reclassifies every affected group as past-changed, whose fresh refold is state-write-free
and reaches the same answer.
"""

from __future__ import annotations

import json
import math

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .run import RunResult, run
from .tables import commit_metadata, commit_metadata_json, read_watermarks, table_exists
from .zset import D_COL

ACC_STATE_SUFFIX = "__trickle_accstate"
_HW, _STATE, _EMPTY = "_hw", "_state", "_empty"


def _q(name: str) -> str:
    return f"`{name}`"


def state_table_for(output: str) -> str:
    return f"{output}{ACC_STATE_SUFFIX}"


# ─── the fold (runs on executors, per group) ────────────────────────────────────


def _fresh(metric_list):
    st = {}
    for out, m in metric_list:
        if m.kind == "sum":
            st[out] = 0
        elif m.kind == "count":
            st[out] = 0
        elif m.kind in ("min", "max", "ema", "product"):
            st[out] = None
        elif m.kind == "first":
            st[out] = [False, None]
        elif m.kind == "tema":
            st[out] = [None, None]  # [value, last along]
        elif m.kind in ("lag", "conv"):
            st[out] = []
        else:  # scan
            st[out] = m.init
    return st


def _step(m, st, out, x, av, row):
    """One fold step: update ``st[out]`` from this row and return the row's output value."""
    if m.kind == "sum":
        st[out] = st[out] + (0 if x is None else x)
        return st[out]
    if m.kind == "count":
        st[out] += 1
        return st[out]
    if m.kind == "min":
        if x is not None and (st[out] is None or x < st[out]):
            st[out] = x
        return st[out]
    if m.kind == "max":
        if x is not None and (st[out] is None or x > st[out]):
            st[out] = x
        return st[out]
    if m.kind == "first":
        if not st[out][0] and x is not None:
            st[out] = [True, x]
        return st[out][1]
    if m.kind == "product":
        if x is not None:
            st[out] = float(x) if st[out] is None else st[out] * float(x)
        return st[out]
    if m.kind == "ema":
        if x is not None:
            st[out] = float(x) if st[out] is None else m.param * float(x) + (1 - m.param) * st[out]
        return st[out]
    if m.kind == "tema":
        if x is not None:
            v, t = st[out]
            if v is None:
                st[out] = [float(x), float(av)]
            else:
                a = 1.0 - math.exp(-m.param * (float(av) - t))
                st[out] = [a * float(x) + (1 - a) * v, float(av)]
        return st[out][0]
    if m.kind == "lag":
        buf = st[out]
        val = buf[0] if len(buf) == m.param else None
        buf.append(x)
        if len(buf) > m.param:
            buf.pop(0)
        return val
    if m.kind == "conv":
        kernel = m.init
        buf = st[out]
        buf.append(0.0 if x is None else float(x))
        if len(buf) > len(kernel):
            buf.pop(0)
        if len(buf) < len(kernel):
            return None
        return float(sum(k * v for k, v in zip(kernel, buf, strict=True)))
    # scan — a custom fold over the whole row
    st[out], val = m.fn(st[out], row)
    return val


def _make_folder(by, along, usercols, metric_list, out_columns):
    """Build the per-group ``applyInPandas`` function. Each group's frame arrives with a ``_state``
    column (the carried JSON, or null for a fresh/past-changed fold); the returned frame carries the
    enriched rows plus, on the **last** row only, the group's new ``_hw``/``_state`` to persist."""
    import pandas as pd

    def fold(pdf):
        pdf = pdf.sort_values(by=[along] + [c for c in usercols if c != along], kind="mergesort")
        raw = pdf[_STATE].iloc[0]
        st = json.loads(raw) if isinstance(raw, str) and raw else _fresh(metric_list)
        outs = {out: [] for out, _ in metric_list}
        hw = None
        for row in pdf[usercols].to_dict("records"):
            row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            av = row[along]
            for out, m in metric_list:
                outs[out].append(_step(m, st, out, row.get(m.col), av, row))
            if av is not None:
                hw = av
        res = pdf[usercols].copy()
        for out, _ in metric_list:
            res[out] = pd.Series(outs[out], index=res.index, dtype="object")
        n = len(res)
        res[_HW] = pd.Series([None] * (n - 1) + [hw], index=res.index, dtype="object")
        res[_STATE] = pd.Series([None] * (n - 1) + [json.dumps(st)], index=res.index, dtype="object")
        return res[out_columns]

    return fold


def _out_type(m, input_types) -> str:
    if m.kind == "count":
        return "bigint"
    if m.kind in ("ema", "tema", "product", "conv"):
        return "double"
    if m.kind == "scan":
        return m.dtype or "double"
    if m.kind == "sum":
        t = input_types[m.col]
        return t if t in ("double", "float", "decimal") or t.startswith("decimal") else "bigint"
    return input_types[m.col]  # min/max/first/lag keep the input column's type


def _run_fold(rows: DataFrame, by, along, metric_list) -> DataFrame:
    """Fold ``rows`` (user cols + ``_state``) per group; returns the enriched frame + state markers."""
    usercols = [c for c in rows.columns if c != _STATE]
    input_types = {f.name: f.dataType.simpleString() for f in rows.schema.fields}
    out_columns = usercols + [out for out, _ in metric_list] + [_HW, _STATE]
    ddl = ", ".join(
        [f"{_q(c)} {input_types[c]}" for c in usercols]
        + [f"{_q(out)} {_out_type(m, input_types)}" for out, m in metric_list]
        + [f"{_q(_HW)} {input_types[along]}", f"{_q(_STATE)} string"]
    )
    folder = _make_folder(by, along, usercols, metric_list, out_columns)
    return rows.groupBy(*[F.col(_q(b)) for b in by]).applyInPandas(folder, schema=ddl)


# ─── state-table I/O ────────────────────────────────────────────────────────────


def _overwrite_state(spark, state: str, df: DataFrame, metadata_json: str) -> None:
    with commit_metadata(spark, metadata_json):
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(state)


def _merge_state(spark, state: str, upserts: DataFrame, by, metadata_json: str) -> None:
    """One MERGE: upsert the folded groups' new state, delete rows flagged ``_empty`` (groups whose
    membership vanished — their carried state must not survive to seed a future revival)."""
    cond = " AND ".join(f"t.{_q(b)} = s.{_q(b)}" for b in by)
    assign = {c: f"s.{_q(c)}" for c in (*by, _HW, _STATE)}
    m = (
        DeltaTable.forName(spark, state)
        .alias("t")
        .merge(upserts.alias("s"), cond)
        .whenMatchedDelete(condition=f"s.{_q(_EMPTY)}")
        .whenMatchedUpdate(set=assign)
        .whenNotMatchedInsert(condition=f"NOT s.{_q(_EMPTY)}", values=assign)
    )
    with commit_metadata(spark, metadata_json):
        m.execute()


# ─── the runner ─────────────────────────────────────────────────────────────────


def run_accumulate(
    spark: SparkSession,
    output: str,
    *,
    by,
    along: str,
    metrics,
    pk,
    sources,
    p,
    compiler_factory,
    ivm: bool = True,
) -> RunResult:
    """One maintenance step of an accumulate plan (see the module docstring for the tail/past split)."""
    metric_list = list(metrics.items())
    state = state_table_for(output)
    need = list(dict.fromkeys([*by, along, *pk, *(m.col for _, m in metric_list if m.col)]))

    def _check(cols, what):
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f".accumulate() into '{output}': the composed {what} is missing column(s) {missing}")

    def full(ctx) -> DataFrame:
        inp = compiler_factory(ctx).current()
        _check(inp.columns, "input")
        folded = _run_fold(inp.withColumn(_STATE, F.lit(None).cast("string")), by, along, metric_list)
        folded = folded.localCheckpoint()
        states = folded.where(F.col(_q(_STATE)).isNotNull()).select(
            *[F.col(_q(b)) for b in by], F.col(_q(_HW)), F.col(_q(_STATE))
        )
        if read_watermarks(spark, state) != ctx.pins:  # replay fast-forward
            _overwrite_state(spark, state, states, commit_metadata_json("accstate", ctx.pins))
        return folded.drop(_HW, _STATE)

    def delta(ctx) -> DataFrame | None:
        if not ivm or not table_exists(spark, state):
            return None
        spins = read_watermarks(spark, state)
        fast_forward = spins == ctx.pins
        if not fast_forward and spins != ctx.last:
            return None  # state drifted from the output's watermarks → comprehensive rebuild
        st = spark.table(state)
        if st.columns != [*by, _HW, _STATE]:
            return None  # the group key changed since the state was built → comprehensive rebuild
        compiler = compiler_factory(ctx)
        zin = compiler.delta()
        if zin is None:
            return None
        _check([c for c in zin.columns if c != D_COL], "input delta")
        zin = zin.cache()

        # classify the affected groups: tail-only (resume) vs past-changed (refold). On a fast-forward
        # the carried state already contains this window — resuming would double-fold — so everything
        # refolds.
        cls = (
            zin.groupBy(*[F.col(_q(b)) for b in by])
            .agg(
                F.expr(f"min({_q(along)})").alias("_mn"),
                F.expr(f"bool_or({_q(D_COL)} < 0)").alias("_ret"),
            )
            .join(st.select(*[F.col(_q(b)) for b in by], F.col(_q(_HW))), on=list(by), how="left")
            .collect()
        )
        if not cls:
            empty = _run_fold(
                compiler.current().limit(0).withColumn(_STATE, F.lit(None).cast("string")), by, along, metric_list
            ).drop(_HW, _STATE)
            return empty.withColumn(D_COL, F.lit(1).cast("long"))
        nby = len(by)
        tail_keys, past_keys = [], []
        for r in cls:
            gk = tuple(r[i] for i in range(nby))
            is_tail = (not fast_forward) and (not r["_ret"]) and (r[_HW] is None or r["_mn"] > r[_HW])
            (tail_keys if is_tail else past_keys).append(gk)

        types = {f.name: f.dataType.simpleString() for f in zin.schema.fields}
        by_schema = ", ".join(f"{_q(b)} {types[b]}" for b in by)  # key tuples are built in `by` order
        parts = []
        if tail_keys:
            tk = spark.createDataFrame(tail_keys, by_schema)
            tail_rows = (
                zin.where(F.col(_q(D_COL)) > 0).drop(D_COL)
                .join(F.broadcast(tk), on=list(by), how="leftsemi")
                .join(st.select(*[F.col(_q(b)) for b in by], F.col(_q(_STATE))), on=list(by), how="left")
            )
            parts.append(tail_rows)
        if past_keys:
            pkdf = spark.createDataFrame(past_keys, by_schema)
            past_rows = (
                compiler.current()
                .join(F.broadcast(pkdf), on=list(by), how="leftsemi")
                .withColumn(_STATE, F.lit(None).cast("string"))
            )
            parts.append(past_rows)
        rows = parts[0]
        for extra in parts[1:]:
            rows = rows.unionByName(extra)

        # localCheckpoint (eager, lineage-truncating), NOT cache: the fold's lineage reads the state
        # table, and the state MERGE below invalidates any cached plan over it — a lazily re-executed
        # fold would then resume from the already-advanced state and double-fold the window.
        folded = _run_fold(rows, by, along, metric_list).localCheckpoint()
        states = folded.where(F.col(_q(_STATE)).isNotNull()).select(
            *[F.col(_q(b)) for b in by], F.col(_q(_HW)), F.col(_q(_STATE)), F.lit(False).alias(_EMPTY)
        )
        if past_keys:  # groups whose membership vanished: no folded rows → flag their state for delete
            pkdf = spark.createDataFrame(past_keys, by_schema)
            emptied = pkdf.join(folded.select(*[F.col(_q(b)) for b in by]).distinct(), on=list(by), how="left_anti")
            emptied = emptied.select(
                *[F.col(_q(b)) for b in by],
                F.lit(None).cast(dict((f.name, f.dataType) for f in folded.schema.fields)[_HW]).alias(_HW),
                F.lit(None).cast("string").alias(_STATE),
                F.lit(True).alias(_EMPTY),
            )
            states = states.unionByName(emptied)
        if not fast_forward:
            _merge_state(spark, state, states, by, commit_metadata_json("accstate", ctx.pins))

        new_out = folded.drop(_HW, _STATE).withColumn(D_COL, F.lit(1).cast("long"))
        if past_keys:  # only past-changed groups have prior rows that may change
            pkdf = spark.createDataFrame(past_keys, by_schema)
            old = spark.table(output).join(F.broadcast(pkdf), on=list(by), how="leftsemi")
            return new_out.unionByName(old.withColumn(D_COL, F.lit(-1).cast("long")))
        return new_out

    return run(spark, output, sources=sources, pk=pk, full=full, delta=delta, p=p)
