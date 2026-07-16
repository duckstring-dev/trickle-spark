"""The builder — a DBSP-style **DAG of binary incremental joins** over Z-set sources.

A fluent, **declarative** plan (ported from duckstring's ``trickle/builder.py``, the reference
implementation): ``ts.source(ref)`` starts a DAG; :meth:`Builder.join` composes another (possibly itself
composed) operand as a binary equi-join of any ``how``; an ordered ``filter``/``mutate``/``select``
output pipeline attaches to the composed result; the terminal :meth:`Builder.merge_into` hands the plan
to :func:`~.run.run`, which owns watermarks, pins, and the commit. Each join node is maintained by the
single **affected-key recompute** rule:

    K = πₖ(δL) ∪ πₖ(δR)                 -- the join-key values that changed on either side
    δO = (O_new restricted to K)(+1) ⊎ (O_old restricted to K)(−1), consolidated

Restricting **both** inputs to ``key ∈ K`` before the join is sound for every join type (it is the
semijoin the join already performs) and is the key pre-filter — a small change never drives a full scan
of the other side; ``K`` is broadcast, which is exactly the plan shape Spark is good at. Re-evaluating
each affected key's full output old-vs-new *is* the match-count logic for the outer incomparables, so
``left``/``right``/``full``/``semi``/``anti`` are maintained the same way as ``inner`` — no privileged
spine, and a bushy ``(A⋈B)⋈(C⋈D)`` composes freely. The K pre-filter uses **null-safe equality**
(``<=>``) so a NULL-keyed changed row still recomputes its (incomparable) output; the join condition
itself stays standard ``=``.

Deviations from the reference, per ``docs/design.md``: the builder holds no context (compile
happens inside the run, against pinned reads); the **old state is time travel** (``VERSION AS OF`` the
watermark), so prior-state reconstruction doesn't exist; nodes compose as lazy DataFrames rather than
temp views (immutable plans — no shared-name rebinding hazard, no ``unique_name``); and the terminals
are :meth:`Builder.merge_into` / :func:`materialize` (chained mid-plan merges are deferred — on a
lakehouse, "materialise an intermediate" is simply another output table with its own ``run()``).
"""

from __future__ import annotations

import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .run import RunContext, RunResult, run
from .zset import D_COL, SYSTEM_PREFIX, consolidate


class BuildError(ValueError):
    """The builder was misconfigured (a missing merge key, an ambiguous join key, a malformed join)."""


# Supported join types → the Spark join-type string. All six are maintained incrementally (per-node
# affected-key recompute), including the outer incomparables, so any of them can sit anywhere in a DAG.
_JOIN_HOW = {"inner": "inner", "left": "left", "right": "right", "full": "full", "semi": "leftsemi", "anti": "leftanti"}
# Join types whose output is the left side only (existence filters) — no right-side columns.
_LEFT_ONLY = {"semi", "anti"}


def _q(name: str) -> str:
    return f"`{name}`"


def normalize_pk(pk) -> tuple[str, ...]:
    if pk is None:
        return ()
    return (pk,) if isinstance(pk, str) else tuple(pk)


def _join_pairs(on) -> list[tuple[str, str]]:
    """Normalise ``on`` to ``[(left_name, right_name), …]``. A str/list names columns shared by both
    sides; a dict maps left columns to right columns. A name may be ``alias.col`` to disambiguate."""
    if isinstance(on, dict):
        return [(left, right) for left, right in on.items()]
    cols = (on,) if isinstance(on, str) else tuple(on)
    return [(c, c) for c in cols]


def _select_items(projection: str) -> list[str]:
    """Split a SQL select list on **top-level** commas (ignoring those inside parens or quotes), so a
    computed item like ``round(a, 2) AS x`` stays one piece."""
    items, depth, buf, quote = [], 0, [], None
    for ch in projection:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"', "`"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        items.append("".join(buf).strip())
    return items


_IDENT = re.compile(r"[A-Za-z_]\w*")
_COLPART = re.compile(r"`[^`]*`|[A-Za-z_]\w*")


def _qualify(text: str, aliases: set[str]) -> str:
    """Rewrite leaf references ``alias.col`` / ``alias.`col``` to the internal qualified column name
    ```alias.col``` (one backticked dotted identifier) — for any ``alias`` in ``aliases``. String
    literals (Spark parses both ``'…'`` and ``"…"`` as strings), backticked identifiers, and unknown
    names pass through untouched."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in ("'", '"'):  # a string literal — copy verbatim (incl. doubled escapes)
            j = i + 1
            while j < n:
                if text[j] == ch:
                    if j + 1 < n and text[j + 1] == ch:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch == "`":  # an already-quoted identifier — copy verbatim
            j = text.find("`", i + 1)
            j = n if j == -1 else j + 1
            out.append(text[i:j])
            i = j
            continue
        m = _IDENT.match(text, i)
        if m:
            word = m.group(0)
            k = m.end()
            if word in aliases and k < n and text[k] == ".":
                cm = _COLPART.match(text, k + 1)
                if cm:
                    col = cm.group(0)
                    bare = col[1:-1] if col.startswith("`") else col
                    out.append(_q(f"{word}.{bare}"))
                    i = cm.end()
                    continue
            out.append(word)
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ─── operator DAG nodes ────────────────────────────────────────────────────────


class _Source:
    """A leaf: a Delta table read as a Z-set source. ``alias`` is the explicit ``.alias()`` name (else a
    positional ``s{i}`` assigned at compile); ``p`` is the change-fraction threshold past which this
    source reads as full."""

    def __init__(self, ref: str, p: float, *, alias: str | None = None) -> None:
        self.ref = ref
        self.p = p
        self.alias = alias

    def leaves(self) -> list["_Source"]:
        return [self]


class _Join:
    """An internal node: a binary equi-join ``left ⋈ right`` of type ``how`` on ``on_pairs`` (raw column
    names, resolved to qualified columns at compile)."""

    def __init__(self, left, right, on_pairs, how: str) -> None:
        self.left = left
        self.right = right
        self.on_pairs = on_pairs
        self.how = how

    def leaves(self) -> list["_Source"]:
        return self.left.leaves() + self.right.leaves()


class _NodeState:
    """One DAG node compiled over this run: ``cols`` (qualified output columns), ``current``/``old``
    (DataFrames for the new and prior full states), ``delta`` (the consolidated Z-set ΔO DataFrame, or
    ``None``), ``changed`` and ``is_full``."""

    __slots__ = ("cols", "current", "old", "delta", "changed", "is_full")

    def __init__(self, cols, current, old, delta, changed, is_full) -> None:
        self.cols = cols
        self.current = current
        self.old = old
        self.delta = delta
        self.changed = changed
        self.is_full = is_full


def source(ref: str, *, p: float = 0.3) -> "Builder":
    """Start a plan at a Delta table. ``p`` is this source's change-fraction threshold (fraction of
    current rows a window may touch before the source reads as full; ``1.0`` disables the check)."""
    return Builder(_Source(ref, p))


class Builder:
    """One handle into the build DAG — declarative; nothing reads a table until the run compiles it."""

    def __init__(self, root) -> None:
        self._root = root
        self._ops: list[tuple[str, object]] = []
        self._alias: str | None = None
        self._sql_query = None  # set by .sql() → comprehensive mode
        self._agg = None  # {"by": (...), "metrics": {out: agg.Metric}} after .aggregate() — terminal-bound
        self._agg_by: tuple[str, ...] | None = None  # staged by .group_by()
        self._along: str | None = None  # the monotonic order axis for .accumulate() (set by .along())
        self._acc = None  # {"by": (...), "metrics": {out: acc.AccMetric}} after .accumulate() — terminal-bound

    # ─── fluent surface ─────────────────────────────────────────────────────────

    def alias(self, name: str) -> "Builder":
        """Name this node: ``.select``/``.filter`` reference its columns by name instead of ``s0``/
        ``s1``, and a following :meth:`sql` uses it as the table name."""
        self._alias = name
        if isinstance(self._root, _Source):
            self._root.alias = name
        return self

    def join(self, dimension: "Builder", *, on, how: str = "inner") -> "Builder":
        """Equi-join another ``ts.source(...)`` operand on ``on`` (a shared column name, a list, or a
        ``{left: right}`` dict; a name may be ``alias.col`` to disambiguate). ``how`` ∈ ``inner``
        (default) / ``left`` / ``right`` / ``full`` / ``semi`` / ``anti`` — all maintained incrementally.

        The operand may itself be a join DAG (bushy/snowflake shapes compose), but must not carry its
        own ``.filter()``/``.mutate()``/``.select()``/``.sql()`` — attach those to the composed result,
        or materialise it as its own output table and source that."""
        self._ensure_composable("join")
        how = how.lower()
        if how not in _JOIN_HOW:
            raise BuildError(f"join(how={how!r}): one of {sorted(_JOIN_HOW)}")
        if not isinstance(dimension, Builder):
            raise BuildError("join() takes another ts.source(...) operand")
        if dimension._sql_query is not None or dimension._agg is not None or dimension._acc is not None:
            raise BuildError(
                "join(): a .sql()/.aggregate()/.accumulate() result can't be a join operand — "
                "materialise it as its own table"
            )
        if dimension._ops:
            raise BuildError(
                "join(): a join operand can't carry its own .filter()/.mutate()/.select() — "
                "attach those to the composed result, or materialise the operand as its own output table"
            )
        self._root = _Join(self._root, dimension._root, _join_pairs(on), how)
        return self

    def filter(self, predicate: str) -> "Builder":
        """Restrict the output with a SQL boolean ``predicate``, evaluated at its **position** in the
        pipeline (call order): a filter placed after a ``.mutate(...)`` may reference the mutated
        column; one before it sees only the source columns."""
        self._ensure_composable("filter")
        self._ops.append(("filter", predicate))
        return self

    def mutate(self, **columns: str) -> "Builder":
        """Add computed columns **without dropping the others** — the ``*``-preserving sibling of
        :meth:`select`. Siblings in one call see the input (not each other); chain calls to build on a
        fresh column; a name matching an existing column replaces it. Expressions must be
        **deterministic** — retractions cancel by full-row identity."""
        self._ensure_composable("mutate")
        if not columns:
            raise BuildError("mutate() needs at least one name=expr column")
        for name in columns:
            if name.startswith(SYSTEM_PREFIX):
                raise BuildError(f"mutate(): column '{name}' uses the reserved '{SYSTEM_PREFIX}' prefix")
        self._ops.append(("mutate", dict(columns)))
        return self

    def select(self, projection: str) -> "Builder":
        """Choose the output column list (a SQL select list), **replacing** the column set at this point
        in the pipeline. Required when a joined DAG's ``*`` would be ambiguous; it must include the
        output PK. Reference columns as ``s{i}`` (left-to-right leaf order) / ``.alias()`` names."""
        self._ensure_composable("select")
        self._ops.append(("select", projection))
        return self

    def sql(self, query) -> "Builder":
        """**The comprehensive escape hatch** — the home for anything outside the incremental op set
        (aggregation until Phase 3, window functions, ``DISTINCT``, set ops, …). The composed result is
        exposed as a temp view named by :meth:`alias` and ``query`` runs over it. It **breaks
        incremental compute but keeps incremental output**: the terminal still diffs the result against
        the current output, so only changed rows reach the table (and its change feed). ``query`` is a
        SQL string, or — with Ibis installed — an Ibis expression compiled lazily."""
        self._ensure_composable("sql")
        if self._alias is None:
            raise BuildError(".sql() needs a table name to reference — call .alias('t') first, then '… FROM t'")
        self._sql_query = query
        return self

    def _ensure_composable(self, op: str) -> None:
        if self._sql_query is not None:
            raise BuildError(
                f".{op}() isn't available after .sql() (the result is materialised, no longer a Z-set) — "
                f"compose joins/filters/projection before .sql()"
            )
        if self._agg is not None:
            raise BuildError(
                f".{op}() can't follow .aggregate() — aggregate is terminal-bound to .merge_into(); do "
                f"further work in a downstream table"
            )
        if self._acc is not None:
            raise BuildError(
                f".{op}() can't follow .accumulate() — the scan is terminal-bound to .merge_into(); do "
                f"further work in a downstream table"
            )

    def along(self, col: str) -> "Builder":
        """Declare the **monotonic order axis** for an order-dependent :meth:`accumulate` scan — a
        column non-decreasing with arrival (a precondition the tail-resume relies on, not a sort of a
        finished result; an out-of-order edit is still handled, by re-folding its group)."""
        self._ensure_composable("along")
        self._along = col
        return self

    def accumulate(self, by=None, **metrics) -> "Builder":
        """Enrich **each** row with order-dependent **running** values — a per-row scan in
        :meth:`along` order partitioned by ``by`` (omit ``by`` for an **ungrouped** scan over the whole
        table — sound, but a single serial fold), using :mod:`trickle_spark.acc` specs. Not a
        reduction (output cardinality = input); terminal-bound to :meth:`merge_into`, which is
        retraction-aware (see ``accumulate.py``). ``pk`` on the terminal is required — the output is
        keyed by row identity, not by the group."""
        self._ensure_composable("accumulate")
        from .acc import AccMetric

        if self._along is None:
            raise BuildError(".accumulate() needs an order axis — call .along('col') first")
        by = normalize_pk(by)
        if not metrics:
            raise BuildError(".accumulate() needs ≥1 metric, e.g. total=acc.sum('qty')")
        for out, m in metrics.items():
            if not isinstance(m, AccMetric):
                raise BuildError(f"accumulate metric '{out}' must be an acc.* spec (acc.sum/count/ema/…)")
        self._acc = {"by": by, "metrics": dict(metrics)}
        return self

    def group_by(self, by) -> "Builder":
        """Ibis-shaped alias: ``.group_by(by).aggregate(**metrics)`` ≡ ``.aggregate(by=by, **metrics)``."""
        self._ensure_composable("group_by")
        self._agg_by = normalize_pk(by)
        return self

    def aggregate(self, by=None, **metrics) -> "Builder":
        """Group the composed output by ``by`` and maintain the ``metrics`` incrementally (see
        ``aggregate.py`` — raw accumulators in a state companion, O(δ) folds, retraction rescans). The
        output ``pk`` defaults to the group key. Terminal-bound to :meth:`merge_into`.

        ``agg.reduce(fn, init)`` metrics are the **order-dependent** exception: they need an
        :meth:`along` axis and can't share an ``.aggregate()`` with order-independent metrics (they run
        on the accumulate machinery — see ``accumulate.py:run_reduce``)."""
        self._ensure_composable("aggregate")
        from .agg import Metric

        if self._agg is not None:
            raise BuildError("one .aggregate() per builder")
        by = normalize_pk(self._agg_by if by is None else by)
        if not by:
            raise BuildError(".aggregate() needs a group key — .aggregate(by=…) or .group_by(…).aggregate(…)")
        if not metrics:
            raise BuildError(".aggregate() needs ≥1 metric, e.g. total=agg.sum('revenue')")
        for out, m in metrics.items():
            if not isinstance(m, Metric):
                raise BuildError(f"aggregate metric '{out}' must be an agg.* spec (e.g. agg.sum/mean/var)")
        if any(m.kind == "reduce" for m in metrics.values()):
            if self._along is None:
                raise BuildError("agg.reduce(...) is order-dependent — call .along('col') before .aggregate()")
            if not all(m.kind == "reduce" for m in metrics.values()):
                raise BuildError("agg.reduce(...) can't share an .aggregate() with other metrics — split them")
        self._agg = {"by": by, "metrics": dict(metrics)}
        return self

    # ─── terminals ──────────────────────────────────────────────────────────────

    def merge_into(self, spark, output: str, *, pk=None, ivm: bool = True, key_filter: bool = True,
                   tag: str | None = None) -> RunResult:
        """Run one maintenance step of this plan into the Delta table ``output`` (see
        :func:`materialize`). ``pk`` is the output identity (must be genuinely unique; after
        :meth:`aggregate` it defaults to the group key).

        ``ivm=False`` ignores deltas and recomputes comprehensively (diffed against the current output);
        ``key_filter=False`` keeps the delta composition but skips the ``key ∈ K`` pre-filter — manual
        escapes, measure before reaching for them."""
        return materialize(spark, output, self, pk=pk, ivm=ivm, key_filter=key_filter, tag=tag)

    def append_to(
        self,
        spark,
        output: str,
        *,
        pk=None,
        fail_on_conflict: bool = True,
        log_drops: bool = True,
        ivm: bool = True,
        key_filter: bool = True,
        tag: str | None = None,
    ) -> RunResult:
        """Run one maintenance step of this plan into the **append-only** Delta table ``output`` — for
        a *monotonic* transform whose output rows are only ever added, never updated or retracted
        (e.g. an append-only fact stream joined to stable dims). New rows are appended in one commit
        carrying the watermarks; nothing is ever updated or deleted, so the table is a true insert-only
        history (its CDF is inserts only).

        An insert-only table can't reflect a change to the past, so two things are **conflicts**: a
        retraction reaching the output, and a ``+1`` row whose ``pk`` is already in the table with a
        *different* image. An identical image is a benign idempotent skip. ``fail_on_conflict=True``
        (default) raises before writing anything; ``False`` drops the conflicting rows (history wins)
        and, with ``log_drops``, records them in a ``{output}__trickle_droplog`` companion. ``pk=None``
        skips the pk checks entirely (only retractions conflict) — fast, sound only when duplicates
        and past-changes are impossible by construction. See ``append.py``."""
        return materialize_append(
            spark, output, self, pk=pk, fail_on_conflict=fail_on_conflict, log_drops=log_drops,
            ivm=ivm, key_filter=key_filter, tag=tag,
        )

    def schema(self, spark) -> dict[str, str]:
        """``{column: Spark type}`` for this plan's output — introspection over the live tables."""
        df = _Compiler(_introspect_ctx(spark), self).current()
        return {f.name: f.dataType.simpleString() for f in df.schema.fields}

    def count(self, spark) -> int:
        """The current row count of this plan's output, computed now over the live tables."""
        return _Compiler(_introspect_ctx(spark), self).current().count()

    # ─── leaf bookkeeping ────────────────────────────────────────────────────────

    def _leaves(self) -> list[_Source]:
        return self._root.leaves()


def materialize(
    spark,
    output: str,
    plan: Builder,
    *,
    pk,
    ivm: bool = True,
    key_filter: bool = True,
    tag: str | None = None,
) -> RunResult:
    """One incremental maintenance step: compile ``plan`` against pinned reads and land one commit on
    ``output`` via :func:`~.run.run` (which owns watermarks, skip/bootstrap/fallback, and the apply)."""
    leaves = plan._leaves()
    sources = list(dict.fromkeys(leaf.ref for leaf in leaves))
    p_map = {leaf.ref: leaf.p for leaf in leaves}

    if plan._acc is not None:
        from .accumulate import run_accumulate

        out_pk = normalize_pk(pk)
        if not out_pk:
            raise BuildError(f"materialize('{output}'): an .accumulate() output needs its row key, pk=...")
        return run_accumulate(
            spark,
            output,
            by=plan._acc["by"],
            along=plan._along,
            metrics=plan._acc["metrics"],
            pk=out_pk,
            sources=sources,
            p=p_map,
            compiler_factory=lambda ctx: _Compiler(ctx, plan, key_filter=key_filter),
            ivm=ivm and plan._sql_query is None,
            tag=tag,
        )

    if plan._agg is not None:
        runner_kwargs = dict(
            by=plan._agg["by"],
            metrics=plan._agg["metrics"],
            pk=normalize_pk(pk) or plan._agg["by"],
            sources=sources,
            p=p_map,
            compiler_factory=lambda ctx: _Compiler(ctx, plan, key_filter=key_filter),
            ivm=ivm and plan._sql_query is None,
            tag=tag,
        )
        if any(m.kind == "reduce" for m in plan._agg["metrics"].values()):
            from .accumulate import run_reduce

            return run_reduce(spark, output, along=plan._along, **runner_kwargs)
        from .aggregate import run_aggregate

        return run_aggregate(spark, output, **runner_kwargs)

    out_pk = normalize_pk(pk)
    if not out_pk:
        raise BuildError(f"materialize('{output}'): pass the output key, pk=...")

    def full(ctx: RunContext) -> DataFrame:
        df = _Compiler(ctx, plan).current()
        _require_pk(out_pk, df.columns, output)
        return df

    def delta(ctx: RunContext) -> DataFrame | None:
        if not ivm or plan._sql_query is not None:
            return None  # comprehensive: run() diffs full() against the current output
        zset = _Compiler(ctx, plan, key_filter=key_filter).delta()
        if zset is not None:
            _require_pk(out_pk, [c for c in zset.columns if c != D_COL], output)
        return zset

    return run(spark, output, sources=sources, pk=out_pk, full=full, delta=delta, p=p_map, tag=tag)


def materialize_append(
    spark,
    output: str,
    plan: Builder,
    *,
    pk,
    fail_on_conflict: bool = True,
    log_drops: bool = True,
    ivm: bool = True,
    key_filter: bool = True,
    tag: str | None = None,
) -> RunResult:
    """One maintenance step landing on an **append-only** output (see :meth:`Builder.append_to`).
    The plan composes exactly as for :func:`materialize`; only the apply differs — a comprehensive
    result is tagged ``+1`` and append-filtered against history rather than diffed, and the run is
    handled by :func:`~.append.run_append`."""
    from .append import run_append

    if plan._agg is not None:
        raise BuildError(".append_to() can't follow .aggregate() — an aggregate updates groups; use .merge_into()")
    if plan._acc is not None:
        raise BuildError(
            ".append_to() can't follow .accumulate() — the scan's terminal is .merge_into() "
            "(retraction-aware; a tail-only stream lands as pure appends through it anyway)"
        )
    out_pk = normalize_pk(pk)
    leaves = plan._leaves()
    sources = list(dict.fromkeys(leaf.ref for leaf in leaves))
    p_map = {leaf.ref: leaf.p for leaf in leaves}

    def full(ctx: RunContext) -> DataFrame:
        df = _Compiler(ctx, plan).current()
        if out_pk:
            _require_pk(out_pk, df.columns, output)
        return df

    def delta(ctx: RunContext) -> DataFrame | None:
        if not ivm or plan._sql_query is not None:
            return None  # comprehensive: run_append tags full() +1 and append-filters it
        zset = _Compiler(ctx, plan, key_filter=key_filter).delta()
        if zset is not None and out_pk:
            _require_pk(out_pk, [c for c in zset.columns if c != D_COL], output)
        return zset

    return run_append(
        spark, output, sources=sources, pk=out_pk, full=full, delta=delta, p=p_map,
        fail_on_conflict=fail_on_conflict, log_drops=log_drops, tag=tag,
    )


def _require_pk(out_pk, cols, output: str) -> None:
    missing = [c for c in out_pk if c not in cols]
    if missing:
        raise BuildError(
            f"the output of '{output}' is missing the PK column(s) {missing} — add them via .select(...)/"
            f".mutate(...), or leave the column in the bare * output"
        )


class _introspect_ctx:
    """An unpinned stand-in for RunContext: live reads, for .schema()/.count() outside a run."""

    def __init__(self, spark) -> None:
        self.spark = spark

    def new(self, ref: str) -> DataFrame:
        return self.spark.table(ref)


# ─── the compiler (per run, against a RunContext) ──────────────────────────────


class _Compiler:
    """Compile one plan over one run's pinned reads. ``current()`` is the comprehensive recompute
    (pipeline applied); ``delta()`` is the composed Z-set ΔO, ``None`` when any source reads full (the
    caller falls back to comprehensive), or an **empty** Z-set when windows moved but nothing changed
    (the run still commits, advancing the watermark via the heartbeat)."""

    def __init__(self, ctx, plan: Builder, *, key_filter: bool = True) -> None:
        self.ctx = ctx
        self.plan = plan
        self.key_filter = key_filter
        self._alias_of: dict[int, str] = {}
        self._cols_cache: dict[int, list[str]] = {}
        self._prepare_leaves()

    def _prepare_leaves(self) -> None:
        seen = set()
        for i, leaf in enumerate(self.plan._leaves()):
            a = leaf.alias or f"s{i}"
            if a in seen:
                raise BuildError(f"duplicate source alias '{a}' — give each ts.source(...) a distinct .alias()")
            seen.add(a)
            self._alias_of[id(leaf)] = a

    def _alias_for(self, leaf: _Source) -> str:
        return self._alias_of[id(leaf)]

    def _bare_cols(self, leaf: _Source) -> list[str]:
        if id(leaf) not in self._cols_cache:
            self._cols_cache[id(leaf)] = list(self.ctx.new(leaf.ref).columns)
        return self._cols_cache[id(leaf)]

    def _aliases_set(self) -> set[str]:
        return set(self._alias_of.values())

    # ─── entry points ────────────────────────────────────────────────────────────

    def current(self) -> DataFrame:
        cols, cur = self._compile_current(self.plan._root)
        out = self._apply_pipeline(cur, cols, is_delta=False)
        if self.plan._sql_query is not None:
            query = self.plan._sql_query
            if not isinstance(query, str):  # an Ibis expression → compile lazily (ibis never a dependency)
                import ibis

                query = str(ibis.to_sql(query, dialect="pyspark"))
            out.createOrReplaceTempView(self.plan._alias)
            out = self.ctx.spark.sql(query)
        return out

    def delta(self) -> DataFrame | None:
        state = self._compile(self.plan._root)
        if state.is_full:
            return None
        if not state.changed:
            # windows moved but consolidated to nothing: an empty ΔO with the right schema, so the run
            # commits the heartbeat and the watermark advances
            empty = self._apply_pipeline(state.current, state.cols, is_delta=False).limit(0)
            return empty.withColumn(D_COL, F.lit(1).cast("long"))
        return self._apply_pipeline(state.delta, state.cols, is_delta=True)

    # ─── compile: current-only (no deltas) ───────────────────────────────────────

    def _qualified(self, leaf: _Source, df: DataFrame, *, keep_d: bool = False):
        """Rename a leaf frame's columns to the internal qualified ``alias.col`` names."""
        a = self._alias_for(leaf)
        bare = [c for c in df.columns if c != D_COL]
        cols = [f"{a}.{c}" for c in bare]
        sel = [F.col(_q(c)).alias(f"{a}.{c}") for c in bare]
        if keep_d:
            sel.append(F.col(_q(D_COL)))
        return cols, df.select(sel)

    def _compile_current(self, node):
        if isinstance(node, _Source):
            return self._qualified(node, self.ctx.new(node.ref))
        lcols, lcur = self._compile_current(node.left)
        rcols, rcur = self._compile_current(node.right)
        return self._join_df(node, lcols, lcur, rcols, rcur)

    def _join_df(self, node: _Join, lcols, lcur, rcols, rcur, *, weight: int | None = None):
        """One join of two frames. ``weight`` (+1/−1) appends the Z-set weight column (a delta term);
        ``None`` is a plain state join. Returns ``(out_cols, DataFrame)``."""
        out_cols = lcols if node.how in _LEFT_ONLY else lcols + rcols
        pairs = self._resolve_pairs(node)
        cond = " AND ".join(f"{_q(lq)} = {_q(rq)}" for lq, rq in pairs)
        joined = lcur.join(rcur, on=F.expr(cond), how=_JOIN_HOW[node.how])
        sel = [F.col(_q(c)) for c in out_cols]
        if weight is not None:
            sel.append(F.lit(int(weight)).cast("long").alias(D_COL))
        return out_cols, joined.select(sel)

    # ─── compile: full (current + old + delta) ───────────────────────────────────

    def _compile(self, node) -> _NodeState:
        if isinstance(node, _Source):
            return self._compile_source(node)
        return self._compile_join(node)

    def _compile_source(self, node: _Source) -> _NodeState:
        cols, current = self._qualified(node, self.ctx.new(node.ref))
        d = self.ctx.delta(node.ref)
        if d.is_full:
            return _NodeState(cols, current, current, None, True, True)
        if d.is_empty():
            return _NodeState(cols, current, current, None, False, False)
        _, delta = self._qualified(node, d.zset, keep_d=True)
        # the old state is time travel at the watermark — the pre-window state by construction
        _, old = self._qualified(node, self.ctx.old(node.ref))
        return _NodeState(cols, current, old, delta, True, False)

    def _compile_join(self, node: _Join) -> _NodeState:
        ls = self._compile(node.left)
        rs = self._compile(node.right)
        out_cols = ls.cols if node.how in _LEFT_ONLY else ls.cols + rs.cols
        is_full = ls.is_full or rs.is_full
        changed = ls.changed or rs.changed
        _, current = self._join_df(node, ls.cols, ls.current, rs.cols, rs.current)
        if is_full or not changed:
            return _NodeState(out_cols, current, current, None, changed, is_full)
        _, old = self._join_df(node, ls.cols, ls.old, rs.cols, rs.old)
        delta = self._join_delta(node, ls, rs, out_cols)
        return _NodeState(out_cols, current, old, delta, True, False)

    def _join_delta(self, node: _Join, ls: _NodeState, rs: _NodeState, out_cols) -> DataFrame:
        """δ(L ⋈ R) by the affected-key recompute. With ``key_filter=False`` the ``K`` restriction is
        skipped — the same diff over the *full* new/old states (correct, just unpruned)."""
        pairs = self._resolve_pairs(node)
        k = self._affected_keys(ls, rs, pairs) if self.key_filter else None
        _, new = self._join_df(node, ls.cols, self._restrict(ls.current, [lq for lq, _ in pairs], k),
                               rs.cols, self._restrict(rs.current, [rq for _, rq in pairs], k), weight=1)
        _, old = self._join_df(node, ls.cols, self._restrict(ls.old, [lq for lq, _ in pairs], k),
                               rs.cols, self._restrict(rs.old, [rq for _, rq in pairs], k), weight=-1)
        return consolidate(new.unionByName(old))

    def _affected_keys(self, ls: _NodeState, rs: _NodeState, pairs) -> DataFrame:
        """The changed join-key values: the left key columns of δL ∪ the right key columns of δR,
        aliased to a common ``k0, k1, …`` and deduplicated."""
        parts = []
        if ls.changed and ls.delta is not None:
            parts.append(ls.delta.select([F.col(_q(lq)).alias(f"k{i}") for i, (lq, _) in enumerate(pairs)]))
        if rs.changed and rs.delta is not None:
            parts.append(rs.delta.select([F.col(_q(rq)).alias(f"k{i}") for i, (_, rq) in enumerate(pairs)]))
        k = parts[0]
        for part in parts[1:]:
            k = k.unionByName(part)
        return k.distinct()

    def _restrict(self, side: DataFrame, keys: list[str], k: DataFrame | None) -> DataFrame:
        """The key pre-filter: a broadcast semi-join of one input against ``K``. **Null-safe** (``<=>``)
        so a NULL-keyed changed row still recomputes its output (an outer join's incomparable)."""
        if k is None:
            return side
        cond = " AND ".join(f"{_q(col)} <=> {_q(f'k{i}')}" for i, col in enumerate(keys))
        return side.join(F.broadcast(k), on=F.expr(cond), how="leftsemi")

    # ─── join-key resolution ──────────────────────────────────────────────────────

    def _resolve_pairs(self, node: _Join):
        out = []
        for lname, rname in node.on_pairs:
            lq = self._resolve_col(node.left, lname, prefer_leftmost=True)
            rq = self._resolve_col(node.right, rname, prefer_leftmost=False)
            out.append((lq, rq))
        return out

    def _resolve_col(self, subtree, name: str, *, prefer_leftmost: bool) -> str:
        """Find the qualified column for ``name`` within ``subtree``: ``alias.col`` (exact) or a bare
        column unique across the subtree's leaves (ties broken by the leftmost leaf only on the left
        side — else ambiguity raises)."""
        if "." in name:
            alias, _, col = name.partition(".")
            for leaf in subtree.leaves():
                if self._alias_for(leaf) == alias and col in self._bare_cols(leaf):
                    return f"{alias}.{col}"
            raise BuildError(f"join key '{name}' not found among the operand's sources")
        hits = [leaf for leaf in subtree.leaves() if name in self._bare_cols(leaf)]
        if not hits:
            raise BuildError(f"join key '{name}' not found among the operand's sources")
        if len(hits) > 1 and not prefer_leftmost:
            aliases = [self._alias_for(leaf) for leaf in hits]
            raise BuildError(f"join key '{name}' is ambiguous across {aliases} — qualify it as 'alias.{name}'")
        return f"{self._alias_for(hits[0])}.{name}"

    # ─── the output pipeline: filter / mutate / select ────────────────────────────
    #
    # The compiled DAG's columns are leaf-qualified ``alias.col``; the pipeline layers over it in call
    # order. Source columns stay qualified through the pipeline; a mutated column is a bare name — the
    # two namespaces never collide (``_qualify`` rewrites ``alias.col`` refs, leaves bare names alone).
    # Every stage is a row-local deterministic map, so applying the pipeline to the Z-set delta is
    # identical to applying it to the full output and re-diffing — incrementally free.

    _STAR_RE = re.compile(r"^(\w+)\.\*$")
    _BARE_RE = re.compile(r"^(\w+)\.(`?)(\w+)\2$")
    _AS_RE = re.compile(r"\bAS\s+(`?)([A-Za-z_]\w*)\1\s*$", re.IGNORECASE)

    @staticmethod
    def _bare_of(col: str) -> str:
        return col.split(".", 1)[1] if "." in col else col

    def _apply_pipeline(self, df: DataFrame, state_cols: list[str], *, is_delta: bool) -> DataFrame:
        aliases = self._aliases_set()
        cols = list(state_cols)  # frame columns: qualified "alias.col" plus bare mutated names
        terminal_select = False
        for kind, payload in self.plan._ops:
            if kind == "filter":
                df = df.where(F.expr(_qualify(payload, aliases)))
                terminal_select = False
            elif kind == "mutate":
                replaced = [c for c in cols if self._bare_of(c) in payload]
                keep = [c for c in cols if c not in replaced]
                sel = [F.col(_q(c)) for c in keep]
                if is_delta:
                    sel.append(F.col(_q(D_COL)))
                sel += [F.expr(_qualify(e, aliases)).alias(n) for n, e in payload.items()]
                df = df.select(sel)
                cols = keep + list(payload)
                terminal_select = False
            else:  # select — replaces the column set with the chosen (bare-named) list
                items, cols = self._select_stage(payload, aliases)
                if is_delta:
                    items = items + [_q(D_COL)]
                df = df.selectExpr(*items)
                terminal_select = True
        if terminal_select:
            return df
        items = self._star_output(cols)
        if is_delta:
            items.append(_q(D_COL))
        return df.selectExpr(*items)

    def _select_stage(self, projection: str, aliases: set[str]):
        """Compile one ``.select`` projection to ``(select_items, output_bare_names)``: ``alias.col`` →
        bare ``col``, ``alias.*`` expands the leaf's columns, a computed item passes through (its name
        from a trailing ``AS``)."""
        out, names = [], []
        for item in _select_items(projection):
            s = item.strip()
            sm = self._STAR_RE.match(s)
            if sm and sm.group(1) in aliases:
                a = sm.group(1)
                leaf = next(leaf for leaf in self.plan._leaves() if self._alias_for(leaf) == a)
                for c in self._bare_cols(leaf):
                    out.append(f"{_q(f'{a}.{c}')} AS {_q(c)}")
                    names.append(c)
                continue
            qd = _qualify(item, aliases)
            bm = self._BARE_RE.match(s)
            if bm and bm.group(1) in aliases:
                out.append(f"{qd} AS {_q(bm.group(3))}")
                names.append(bm.group(3))
            else:
                out.append(qd)
                am = self._AS_RE.search(s)
                names.append(am.group(2) if am else s)  # best-effort name (used only if a stage follows)
        return out, names

    def _join_key_finder(self):
        """Union-find over the qualified columns equated by every ``on=`` in the DAG, so the ``*``
        output can deduplicate equi-join keys (same value on both sides) while still rejecting any other
        name collision."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def walk(node) -> None:
            if isinstance(node, _Join):
                for lq, rq in self._resolve_pairs(node):
                    parent[find(lq)] = find(rq)
                walk(node.left)
                walk(node.right)

        walk(self.plan._root)
        return find

    def _star_output(self, cols: list[str]) -> list[str]:
        """Bare-name the surviving frame columns for the implicit ``*``: equi-join keys colliding on the
        bare name fold to the leftmost copy; any other bare-name collision raises."""
        find = self._join_key_finder()
        by_bare: dict[str, list[str]] = {}
        for c in cols:
            by_bare.setdefault(self._bare_of(c), []).append(c)
        out = []
        for bare, members in by_bare.items():
            if len(members) > 1 and len({find(m) for m in members}) != 1:
                where = [m.split(".", 1)[0] for m in members if "." in m] or members
                raise BuildError(
                    f"column '{bare}' is ambiguous across {where} — name the survivors with .select(...) or "
                    f"rename via .mutate(); only equi-join keys are auto-deduplicated"
                )
            c = members[0]  # cols is left-to-right (leaf order), so this is the leftmost copy
            out.append(_q(c) if "." not in c else f"{_q(c)} AS {_q(bare)}")
        return out
