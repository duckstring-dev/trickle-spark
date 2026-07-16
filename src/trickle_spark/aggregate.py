"""Incremental grouped aggregation: the ``.aggregate(by, **metrics)`` machinery.

Raw accumulators live in a **state companion** Delta table ``{output}__trickle_aggstate`` (per additive
column a running sum, non-NULL count and centred moment M2; per extreme a stored min & max; per
weighted unit Σ(w·x), Σw and a pair count; per argmin/argmax the supporting key + payload; per
semigroup the reduced value; per product the sign/zero counts + Σ log|x|). The published output holds
only the derived user columns — a clean table like any other trickle-spark output.

**Incremental path** (an input Z-set ΔO): fold the distributive sums additively per affected group
(O(δ)); maintain M2 by the parallel **Chan/Pébay merge-in/merge-out** of centred moments computed
about the delta partitions' own means (two passes — numerically well-conditioned, never
``Σx² − (Σx)²/n``); the extend-or-rescan families (min/max/arg/bool/bit) extend in place from the
inserts, but a group with **any retraction rescans its current membership** (the supporting row may be
gone). Emit ``new (+1) ⊎ old (−1)`` for the affected groups — old read from the **current output
table** (materialised — no reconstruction). **Comprehensive path**: rebuild the accumulators wholesale
and let the run diff the derived output. A group whose count reaches 0 is dropped and retracted.

**Two-table atomicity** (the one seam in the no-control-plane story, per ``docs/design.md``):
the state companion and the output cannot commit together, so the order is **state first, output
second**, each commit carrying the same pinned-version map. A re-run after a crash-between sees the
state already at the run's pins and **fast-forwards** — the state write is skipped and the output delta
is derived from the (already-updated) state against the (not-yet-updated) output. Any other divergence
between the state's pins and the output's watermarks falls back to a comprehensive rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .run import RunResult, run
from .tables import commit_metadata, commit_metadata_json, read_watermarks, table_exists
from .zset import D_COL

AGG_STATE_SUFFIX = "__trickle_aggstate"

# Families whose stored value can be invalidated by a retraction (the supporting row may be the one
# retracted) — a group with any retraction rescans its current membership; append-only groups never do.
RESCAN_KINDS = {"min", "max", "argmin", "argmax", "bool_and", "bool_or", "bit_and", "bit_or"}

_SG_FN = {"bool_and": "bool_and", "bool_or": "bool_or", "bit_and": "bit_and", "bit_or": "bit_or"}
_SG_OP = {"bool_and": "AND", "bool_or": "OR", "bit_and": "&", "bit_or": "|"}


def _q(name: str) -> str:
    return f"`{name}`"


@dataclass
class _Families:
    add_cols: list  # sum/mean/var/stddev input columns (Σx, non-NULL count, M2 each)
    ext_cols: list  # min/max input columns
    wgt_units: list  # (x | None, w) pairs — weighted_sum/weighted_average, or weight_total (x=None)
    arg_specs: list  # (arg, key, "min" | "max")
    sg_specs: list  # (col, kind) for bool_and/bool_or/bit_and/bit_or
    prod_cols: list  # product input columns

    @property
    def needs_rescan(self) -> bool:
        return bool(self.ext_cols or self.arg_specs or self.sg_specs)


def classify(metrics) -> _Families:
    add, ext, wgt, arg, sg, prod = [], [], [], [], [], []

    def put(bucket, item):
        if item not in bucket:
            bucket.append(item)

    for m in metrics.values():
        if m.kind in ("sum", "mean", "var", "stddev"):
            put(add, m.col)
        elif m.kind in ("min", "max"):
            put(ext, m.col)
        elif m.kind == "weight_total":
            put(wgt, (None, m.col))
        elif m.kind in ("weighted_sum", "weighted_average"):
            put(wgt, (m.col, m.col2))
        elif m.kind in ("argmin", "argmax"):
            put(arg, (m.col, m.col2, m.kind[3:]))
        elif m.kind in _SG_FN:
            put(sg, (m.col, m.kind))
        elif m.kind == "product":
            put(prod, m.col)
        elif m.kind != "count":
            raise ValueError(f"agg metric kind {m.kind!r} is not supported by trickle-spark yet")
    return _Families(add, ext, wgt, arg, sg, prod)


def required_columns(by, metrics) -> list[str]:
    need = list(by)
    for m in metrics.values():
        for c in (m.col, m.col2):
            if c is not None and c not in need:
                need.append(c)
    return need


def acc_columns(fams: _Families) -> list[str]:
    cols = ["_a_cnt"]
    for i in range(len(fams.add_cols)):
        cols += [f"_a_sum_{i}", f"_a_cnt_{i}", f"_a_m2_{i}"]
    for j in range(len(fams.ext_cols)):
        cols += [f"_a_min_{j}", f"_a_max_{j}"]
    for m in range(len(fams.wgt_units)):
        cols += [f"_w_num_{m}", f"_w_den_{m}", f"_w_cnt_{m}"]
    for a in range(len(fams.arg_specs)):
        cols += [f"_g_key_{a}", f"_g_arg_{a}"]
    for g in range(len(fams.sg_specs)):
        cols += [f"_s_val_{g}"]
    for p in range(len(fams.prod_cols)):
        cols += [f"_p_cnt_{p}", f"_p_nz_{p}", f"_p_nn_{p}", f"_p_sl_{p}"]
    return cols


# ─── accumulator expressions ────────────────────────────────────────────────────


def rebuild_state(inp: DataFrame, by, fams: _Families) -> DataFrame:
    """The comprehensive accumulator rebuild: one aggregation over the full clean input."""
    exprs = ["CAST(count(*) AS BIGINT) AS _a_cnt"]
    for i, c in enumerate(fams.add_cols):
        exprs += [
            f"sum({_q(c)}) AS _a_sum_{i}",
            f"CAST(count({_q(c)}) AS BIGINT) AS _a_cnt_{i}",
            f"CAST(coalesce(var_pop({_q(c)}) * count({_q(c)}), 0.0) AS DOUBLE) AS _a_m2_{i}",
        ]
    for j, c in enumerate(fams.ext_cols):
        exprs += [f"min({_q(c)}) AS _a_min_{j}", f"max({_q(c)}) AS _a_max_{j}"]
    for m, (x, w) in enumerate(fams.wgt_units):
        both = f"{_q(w)} IS NOT NULL" if x is None else f"{_q(x)} IS NOT NULL AND {_q(w)} IS NOT NULL"
        num = "CAST(0.0 AS DOUBLE)" if x is None else f"coalesce(sum(CASE WHEN {both} THEN {_q(w)} * {_q(x)} END), 0.0)"
        exprs += [
            f"CAST({num} AS DOUBLE) AS _w_num_{m}",
            f"CAST(coalesce(sum(CASE WHEN {both} THEN {_q(w)} END), 0.0) AS DOUBLE) AS _w_den_{m}",
            f"CAST(coalesce(sum(CASE WHEN {both} THEN 1 ELSE 0 END), 0) AS BIGINT) AS _w_cnt_{m}",
        ]
    for a, (arg, key, d) in enumerate(fams.arg_specs):
        exprs += [f"{d}({_q(key)}) AS _g_key_{a}", f"{d}_by({_q(arg)}, {_q(key)}) AS _g_arg_{a}"]
    for g, (c, kind) in enumerate(fams.sg_specs):
        exprs += [f"{_SG_FN[kind]}({_q(c)}) AS _s_val_{g}"]
    for p, c in enumerate(fams.prod_cols):
        nn = f"{_q(c)} IS NOT NULL"
        exprs += [
            f"CAST(count({_q(c)}) AS BIGINT) AS _p_cnt_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} = 0 THEN 1 ELSE 0 END), 0) AS BIGINT) AS _p_nz_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} < 0 THEN 1 ELSE 0 END), 0) AS BIGINT) AS _p_nn_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} <> 0 THEN ln(abs({_q(c)})) END), 0.0) AS DOUBLE) AS _p_sl_{p}",
        ]
    return inp.groupBy(*[F.col(_q(b)) for b in by]).agg(*[F.expr(e) for e in exprs])


def fold_delta(zset: DataFrame, by, fams: _Families) -> DataFrame:
    """Per-group accumulator deltas from an input Z-set ΔO (two passes for the centred moments: the
    insert/delete partitions' sums first, then their central moments about their own means)."""
    d = _q(D_COL)
    by_cols = [F.col(_q(b)) for b in by]
    exprs = [f"CAST(sum({d}) AS BIGINT) AS _a_cnt"]
    for i, c in enumerate(fams.add_cols):
        nn = f"{_q(c)} IS NOT NULL"
        exprs += [
            f"CAST(coalesce(sum(CASE WHEN {d} > 0 AND {nn} THEN {d} ELSE 0 END), 0) AS BIGINT) AS _dni_{i}",
            f"coalesce(sum(CASE WHEN {d} > 0 AND {nn} THEN {d} * {_q(c)} END), 0) AS _dsi_{i}",
            f"CAST(coalesce(sum(CASE WHEN {d} < 0 AND {nn} THEN -{d} ELSE 0 END), 0) AS BIGINT) AS _dnd_{i}",
            f"coalesce(sum(CASE WHEN {d} < 0 AND {nn} THEN -{d} * {_q(c)} END), 0) AS _dsd_{i}",
        ]
    for j, c in enumerate(fams.ext_cols):
        exprs += [
            f"min(CASE WHEN {d} > 0 THEN {_q(c)} END) AS _e_minp_{j}",
            f"max(CASE WHEN {d} > 0 THEN {_q(c)} END) AS _e_maxp_{j}",
        ]
    for m, (x, w) in enumerate(fams.wgt_units):
        both = f"{_q(w)} IS NOT NULL" if x is None else f"{_q(x)} IS NOT NULL AND {_q(w)} IS NOT NULL"
        num = "CAST(0.0 AS DOUBLE)" if x is None \
            else f"coalesce(sum(CASE WHEN {both} THEN {d} * {_q(w)} * {_q(x)} END), 0.0)"
        exprs += [
            f"CAST({num} AS DOUBLE) AS _dw_num_{m}",
            f"CAST(coalesce(sum(CASE WHEN {both} THEN {d} * {_q(w)} END), 0.0) AS DOUBLE) AS _dw_den_{m}",
            f"CAST(coalesce(sum(CASE WHEN {both} THEN {d} ELSE 0 END), 0) AS BIGINT) AS _dw_cnt_{m}",
        ]
    for a, (arg, key, dr) in enumerate(fams.arg_specs):
        exprs += [
            f"{dr}(CASE WHEN {d} > 0 THEN {_q(key)} END) AS _dg_key_{a}",
            f"{dr}_by(CASE WHEN {d} > 0 THEN {_q(arg)} END, CASE WHEN {d} > 0 THEN {_q(key)} END) AS _dg_arg_{a}",
        ]
    for g, (c, kind) in enumerate(fams.sg_specs):
        exprs += [f"{_SG_FN[kind]}(CASE WHEN {d} > 0 THEN {_q(c)} END) AS _ds_val_{g}"]
    for p, c in enumerate(fams.prod_cols):
        nn = f"{_q(c)} IS NOT NULL"
        exprs += [
            f"CAST(coalesce(sum(CASE WHEN {nn} THEN {d} ELSE 0 END), 0) AS BIGINT) AS _dp_cnt_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} = 0 THEN {d} ELSE 0 END), 0) AS BIGINT) AS _dp_nz_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} < 0 THEN {d} ELSE 0 END), 0) AS BIGINT) AS _dp_nn_{p}",
            f"CAST(coalesce(sum(CASE WHEN {nn} AND {_q(c)} <> 0 THEN {d} * ln(abs({_q(c)})) END), 0.0) AS DOUBLE)"
            f" AS _dp_sl_{p}",
        ]
    if fams.needs_rescan:
        exprs.append(f"bool_or({d} < 0) AS _a_ret")
    pass1 = zset.groupBy(*by_cols).agg(*[F.expr(e) for e in exprs])

    if not fams.add_cols:
        return pass1
    # Second pass: each partition's central moment about its own mean (deviations are O(spread), not
    # O(value) — the well-conditioned form). Joined back onto pass1.
    joined = zset.join(pass1, on=list(by), how="inner")
    mexprs = []
    for i, c in enumerate(fams.add_cols):
        nn = f"{_q(c)} IS NOT NULL"
        mexprs += [
            f"CAST(coalesce(sum(CASE WHEN {d} > 0 AND {nn} AND _dni_{i} > 0 "
            f"THEN {d} * pow({_q(c)} - _dsi_{i} / _dni_{i}, 2) END), 0.0) AS DOUBLE) AS _dm2i_{i}",
            f"CAST(coalesce(sum(CASE WHEN {d} < 0 AND {nn} AND _dnd_{i} > 0 "
            f"THEN -{d} * pow({_q(c)} - _dsd_{i} / _dnd_{i}, 2) END), 0.0) AS DOUBLE) AS _dm2d_{i}",
        ]
    pass2 = joined.groupBy(*by_cols).agg(*[F.expr(e) for e in mexprs])
    return pass1.join(pass2, on=list(by), how="left")


def rescan_groups(current: DataFrame, ret_groups: DataFrame, by, fams: _Families) -> DataFrame:
    """Recompute the rescan families over the **current membership** of the retracting groups."""
    exprs = []
    for j, c in enumerate(fams.ext_cols):
        exprs += [f"min({_q(c)}) AS _r_min_{j}", f"max({_q(c)}) AS _r_max_{j}"]
    for a, (arg, key, d) in enumerate(fams.arg_specs):
        exprs += [f"{d}({_q(key)}) AS _r_key_{a}", f"{d}_by({_q(arg)}, {_q(key)}) AS _r_arg_{a}"]
    for g, (c, kind) in enumerate(fams.sg_specs):
        exprs += [f"{_SG_FN[kind]}({_q(c)}) AS _r_val_{g}"]
    scoped = current.join(F.broadcast(ret_groups), on=list(by), how="leftsemi")
    return scoped.groupBy(*[F.col(_q(b)) for b in by]).agg(*[F.expr(e) for e in exprs])


def merge_rows(state: DataFrame, dacc: DataFrame, rescan: DataFrame | None, by, fams: _Families) -> DataFrame:
    """The affected groups' **new** accumulator rows: state ⟕ dacc (⟕ rescan), folded additively — with
    M2 by the Chan/Pébay merge-in (insert partition) then merge-out (delete partition)."""
    joined = dacc.alias("d").join(state.alias("a"), on=list(by), how="left")
    if rescan is not None:
        joined = joined.join(rescan.alias("r"), on=list(by), how="left")
    ret = "coalesce(d._a_ret, false)" if fams.needs_rescan else "false"

    def r(col: str) -> str:
        # the rescan frame exists only when some group actually retracted; when it doesn't, the CASE's
        # rescan branch is never taken but its reference must still resolve — substitute NULL
        return f"r.{col}" if rescan is not None else "NULL"

    sel = ["CAST(coalesce(a._a_cnt, 0) + d._a_cnt AS BIGINT) AS _a_cnt"]
    for i in range(len(fams.add_cols)):
        n0, s0, m0 = f"coalesce(a._a_cnt_{i}, 0)", f"coalesce(a._a_sum_{i}, 0)", f"coalesce(a._a_m2_{i}, 0.0)"
        ni, si, mi = f"d._dni_{i}", f"d._dsi_{i}", f"coalesce(d._dm2i_{i}, 0.0)"
        nd, sd, md = f"d._dnd_{i}", f"d._dsd_{i}", f"coalesce(d._dm2d_{i}, 0.0)"
        nc = f"({n0} + {ni})"
        mc = (
            f"(CASE WHEN {ni} = 0 THEN {m0} WHEN {n0} = 0 THEN {mi} "
            f"ELSE {m0} + {mi} + (CAST({n0} AS DOUBLE) * {ni} / {nc}) * pow({s0} / {n0} - {si} / {ni}, 2) END)"
        )
        nf = f"({nc} - {nd})"
        sf = f"({s0} + {si} - {sd})"
        mf = (
            f"(CASE WHEN {nd} = 0 THEN {mc} WHEN {nf} <= 0 THEN 0.0 "
            f"ELSE greatest({mc} - {md} - (CAST({nd} AS DOUBLE) * {nf} / {nc}) * pow({sd} / {nd} - {sf} / {nf}, 2),"
            f" 0.0) END)"
        )
        sel += [f"{sf} AS _a_sum_{i}", f"CAST({nf} AS BIGINT) AS _a_cnt_{i}", f"CAST({mf} AS DOUBLE) AS _a_m2_{i}"]
    for j in range(len(fams.ext_cols)):
        sel += [
            f"CASE WHEN {ret} THEN {r(f'_r_min_{j}')} ELSE least(a._a_min_{j}, d._e_minp_{j}) END AS _a_min_{j}",
            f"CASE WHEN {ret} THEN {r(f'_r_max_{j}')} ELSE greatest(a._a_max_{j}, d._e_maxp_{j}) END AS _a_max_{j}",
        ]
    for m in range(len(fams.wgt_units)):
        sel += [
            f"CAST(coalesce(a._w_num_{m}, 0.0) + d._dw_num_{m} AS DOUBLE) AS _w_num_{m}",
            f"CAST(coalesce(a._w_den_{m}, 0.0) + d._dw_den_{m} AS DOUBLE) AS _w_den_{m}",
            f"CAST(coalesce(a._w_cnt_{m}, 0) + d._dw_cnt_{m} AS BIGINT) AS _w_cnt_{m}",
        ]
    for a, (_arg, _key, dr) in enumerate(fams.arg_specs):
        better = "<" if dr == "min" else ">"
        pick = "least" if dr == "min" else "greatest"
        sel += [
            f"CASE WHEN {ret} THEN {r(f'_r_key_{a}')} ELSE {pick}(a._g_key_{a}, d._dg_key_{a}) END AS _g_key_{a}",
            f"CASE WHEN {ret} THEN {r(f'_r_arg_{a}')} "
            f"WHEN d._dg_key_{a} IS NOT NULL AND (a._g_key_{a} IS NULL OR d._dg_key_{a} {better} a._g_key_{a}) "
            f"THEN d._dg_arg_{a} ELSE a._g_arg_{a} END AS _g_arg_{a}",
        ]
    for g, (_c, kind) in enumerate(fams.sg_specs):
        op = _SG_OP[kind]
        sel += [
            f"CASE WHEN {ret} THEN {r(f'_r_val_{g}')} "
            f"WHEN a._s_val_{g} IS NULL THEN d._ds_val_{g} WHEN d._ds_val_{g} IS NULL THEN a._s_val_{g} "
            f"ELSE a._s_val_{g} {op} d._ds_val_{g} END AS _s_val_{g}",
        ]
    for p in range(len(fams.prod_cols)):
        sel += [
            f"CAST(coalesce(a._p_cnt_{p}, 0) + d._dp_cnt_{p} AS BIGINT) AS _p_cnt_{p}",
            f"CAST(coalesce(a._p_nz_{p}, 0) + d._dp_nz_{p} AS BIGINT) AS _p_nz_{p}",
            f"CAST(coalesce(a._p_nn_{p}, 0) + d._dp_nn_{p} AS BIGINT) AS _p_nn_{p}",
            f"CAST(coalesce(a._p_sl_{p}, 0.0) + d._dp_sl_{p} AS DOUBLE) AS _p_sl_{p}",
        ]
    return joined.selectExpr(*[f"d.{_q(b)} AS {_q(b)}" for b in by], *sel)


def derive(state: DataFrame, by, metrics, fams: _Families) -> DataFrame:
    """The published user columns from the accumulators (groups with rows only)."""
    sidx = {c: i for i, c in enumerate(fams.add_cols)}
    eidx = {c: j for j, c in enumerate(fams.ext_cols)}
    widx = {u: m for m, u in enumerate(fams.wgt_units)}
    aidx = {s: a for a, s in enumerate(fams.arg_specs)}
    gidx = {s: g for g, s in enumerate(fams.sg_specs)}
    pidx = {c: p for p, c in enumerate(fams.prod_cols)}
    exprs = []
    for out, m in metrics.items():
        if m.kind == "count":
            e = "_a_cnt"
        elif m.kind == "sum":
            i = sidx[m.col]
            e = f"CASE WHEN _a_cnt_{i} > 0 THEN _a_sum_{i} END"
        elif m.kind == "mean":
            i = sidx[m.col]
            e = f"CASE WHEN _a_cnt_{i} > 0 THEN _a_sum_{i} / _a_cnt_{i} END"
        elif m.kind in ("var", "stddev"):
            i = sidx[m.col]
            v = f"_a_m2_{i} / (_a_cnt_{i} - 1)" if m.how == "sample" else f"_a_m2_{i} / _a_cnt_{i}"
            lo = 2 if m.how == "sample" else 1
            e = f"CASE WHEN _a_cnt_{i} >= {lo} THEN {v} END"
            if m.kind == "stddev":
                e = f"sqrt({e})"
        elif m.kind in ("min", "max"):
            e = f"_a_{m.kind}_{eidx[m.col]}"
        elif m.kind == "weight_total":
            mm = widx[(None, m.col)]
            e = f"CASE WHEN _w_cnt_{mm} > 0 THEN _w_den_{mm} END"
        elif m.kind == "weighted_sum":
            mm = widx[(m.col, m.col2)]
            e = f"CASE WHEN _w_cnt_{mm} > 0 THEN _w_num_{mm} END"
        elif m.kind == "weighted_average":
            mm = widx[(m.col, m.col2)]
            e = f"CASE WHEN _w_cnt_{mm} > 0 AND _w_den_{mm} <> 0 THEN _w_num_{mm} / _w_den_{mm} END"
        elif m.kind in ("argmin", "argmax"):
            e = f"_g_arg_{aidx[(m.col, m.col2, m.kind[3:])]}"
        elif m.kind in _SG_FN:
            e = f"_s_val_{gidx[(m.col, m.kind)]}"
        else:  # product
            p = pidx[m.col]
            e = (
                f"CASE WHEN _p_cnt_{p} = 0 THEN CAST(NULL AS DOUBLE) WHEN _p_nz_{p} > 0 THEN 0.0 "
                f"ELSE (CASE WHEN _p_nn_{p} % 2 = 1 THEN -1.0 ELSE 1.0 END) * exp(_p_sl_{p}) END"
            )
        exprs.append(f"{e} AS {_q(out)}")
    return state.where("_a_cnt > 0").selectExpr(*[_q(b) for b in by], *exprs)


# ─── state-table I/O (state first, output second — see the module docstring) ───


def state_table_for(output: str) -> str:
    return f"{output}{AGG_STATE_SUFFIX}"


def _overwrite_state(spark, state: str, df: DataFrame, metadata_json: str) -> None:
    with commit_metadata(spark, metadata_json):
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(state)


def _merge_state(spark, state: str, merged: DataFrame, by, metadata_json: str) -> None:
    cond = " AND ".join(f"t.{_q(b)} = s.{_q(b)}" for b in by)
    assign = {c: f"s.{_q(c)}" for c in merged.columns}
    m = (
        DeltaTable.forName(spark, state)
        .alias("t")
        .merge(merged.alias("s"), cond)
        .whenMatchedDelete(condition="s._a_cnt <= 0")
        .whenMatchedUpdate(set=assign)
        .whenNotMatchedInsert(condition="s._a_cnt > 0", values=assign)
    )
    with commit_metadata(spark, metadata_json):
        m.execute()


# ─── the runner ─────────────────────────────────────────────────────────────────


def run_aggregate(
    spark: SparkSession,
    output: str,
    *,
    by,
    metrics,
    pk,
    sources,
    p,
    compiler_factory,
    ivm: bool = True,
) -> RunResult:
    """One maintenance step of an aggregate plan. ``compiler_factory(ctx)`` yields the plan compiler
    (``.current()`` = the composed input rows, ``.delta()`` = the input ΔO); everything else — the
    watermark ladder, the single output commit — is :func:`~.run.run`."""
    fams = classify(metrics)
    state = state_table_for(output)
    need = required_columns(by, metrics)

    def _check(cols, what):
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f".aggregate() into '{output}': the composed {what} is missing column(s) {missing}")

    def full(ctx) -> DataFrame:
        inp = compiler_factory(ctx).current()
        _check(inp.columns, "input")
        meta = commit_metadata_json("aggstate", ctx.pins)
        if read_watermarks(spark, state) != ctx.pins:  # replay fast-forward: state already at these pins
            _overwrite_state(spark, state, rebuild_state(inp, by, fams), meta)
        return derive(spark.table(state), by, metrics, fams)

    def delta(ctx) -> DataFrame | None:
        if not ivm or not table_exists(spark, state):
            return None
        spins = read_watermarks(spark, state)
        fast_forward = spins == ctx.pins
        if not fast_forward and spins != ctx.last:
            return None  # state drifted from the output's watermarks → comprehensive rebuild
        expected = list(by) + acc_columns(fams)
        if spark.table(state).columns != expected:
            return None  # the metric set changed since the state was built → comprehensive rebuild
        compiler = compiler_factory(ctx)
        zin = compiler.delta()
        if zin is None:
            return None
        _check([c for c in zin.columns if c != D_COL], "input delta")
        zin = zin.cache()
        dacc = fold_delta(zin, by, fams).cache()
        if not dacc.take(1):
            empty = derive(spark.table(state).limit(0), by, metrics, fams)
            return empty.withColumn(D_COL, F.lit(1).cast("long"))
        rescan = None
        if fams.needs_rescan:
            ret_groups = dacc.where("_a_ret").select(*[F.col(_q(b)) for b in by])
            if ret_groups.take(1):
                rescan = rescan_groups(compiler.current(), ret_groups, by, fams)
        if not fast_forward:
            merged = merge_rows(spark.table(state), dacc, rescan, by, fams)
            _merge_state(spark, state, merged, by, commit_metadata_json("aggstate", ctx.pins))
        affected = dacc.select(*[F.col(_q(b)) for b in by])
        new_out = derive(spark.table(state).join(affected, on=list(by), how="leftsemi"), by, metrics, fams)
        old_out = spark.table(output).join(affected, on=list(by), how="leftsemi")
        return (
            new_out.withColumn(D_COL, F.lit(1).cast("long"))
            .unionByName(old_out.withColumn(D_COL, F.lit(-1).cast("long")))
        )

    return run(spark, output, sources=sources, pk=pk, full=full, delta=delta, p=p)
