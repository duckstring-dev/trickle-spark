"""Scan (order-dependent) metric specs for the builder's ``.along(...).accumulate(by=…, name=acc.…)``.

Where :mod:`trickle_spark.agg` reductions are **order-independent** (one value per group), an ``acc.*``
scan is **order-dependent** and **per-row**: it enriches *every* row with its running value computed in
the ``.along(...)`` order within its ``by`` group. The output has the same cardinality as the input —
``.accumulate()`` is a transform, not a reduction — finished by :meth:`~.builder.Builder.merge_into`,
which is **retraction-aware**: a tail-only append resumes each group's carried fold-state (O(new)); a
retraction or edit in a group's past re-folds that group over its current membership (O(group)) and
merge-diffs, so only rows whose running values actually changed reach the output's change feed.

The fold itself runs as ``applyInPandas`` per group (every metric — including the recursive ``ema`` /
``tema`` and the FIFO-buffer ``lag``/``convolution``/``scan`` — handled uniformly); carried state is
JSON-persisted per group in a ``{output}__trickle_accstate`` companion, so it must stay
JSON-serializable (numbers/strings/lists — a tuple round-trips to a list).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccMetric:
    """One scan output: its ``kind``, the input column it reads (if any), the scalar parameter
    (``alpha`` for ema / ``lam`` for tema / ``n`` for lag), and — for :func:`scan` /
    :func:`convolution` — the reducer/kernel and output ``dtype`` (a Spark type string)."""

    kind: str
    col: str | None = None
    param: float | None = None
    fn: object = None
    init: object = None
    dtype: str | None = None


def sum(col: str) -> AccMetric:  # noqa: A001 - deliberate SQL-style name on the acc namespace
    """Running sum of ``col`` along the axis (NULLs contribute 0)."""
    return AccMetric("sum", col)


def count() -> AccMetric:
    """Running count of rows seen so far in the group (1, 2, 3, …)."""
    return AccMetric("count")


def min(col: str) -> AccMetric:  # noqa: A001 - deliberate SQL-style name on the acc namespace
    """Running minimum of ``col`` seen so far (NULLs ignored)."""
    return AccMetric("min", col)


def max(col: str) -> AccMetric:  # noqa: A001 - deliberate SQL-style name on the acc namespace
    """Running maximum of ``col`` seen so far (NULLs ignored)."""
    return AccMetric("max", col)


def first(col: str) -> AccMetric:
    """The first non-NULL value of ``col`` in the group — frozen once set, emitted on every later row."""
    return AccMetric("first", col)


def product(col: str) -> AccMetric:  # noqa: A001 - mirrors agg.product on the scan namespace
    """Running product of ``col`` (NULLs ignored; the first non-NULL seeds it; a 0 keeps it at 0).
    Output is a DOUBLE — large products overflow to ±inf."""
    return AccMetric("product", col)


def ema(col: str, alpha: float) -> AccMetric:
    """Discrete exponential moving average — ``α·x + (1−α)·ema_prev`` per row, ``0 < α ≤ 1``."""
    if not 0 < alpha <= 1:
        raise ValueError(f"ema(alpha={alpha!r}): need 0 < alpha <= 1")
    return AccMetric("ema", col, float(alpha))


def tema(col: str, lam: float) -> AccMetric:
    """Time-decayed (continuous) EMA whose decay scales with the gap ``Δt`` in the ``.along`` value
    (which must be numeric): ``α_t = 1 − exp(−lam·Δt)``, ``lam > 0``."""
    if lam <= 0:
        raise ValueError(f"tema(lam={lam!r}): need lam > 0")
    return AccMetric("tema", col, float(lam))


def prev(col: str) -> AccMetric:
    """The value of ``col`` one row back in the group (``lag`` 1) — NULL on the first row. Reaches
    across run boundaries (the one-slot buffer is carried state)."""
    return AccMetric("lag", col, 1)


def lag(col: str, n: int = 1) -> AccMetric:
    """The value of ``col`` ``n`` rows back in the group (NULL until the group has ``n`` prior rows).
    Carried as a length-``n`` FIFO buffer, so it reaches back across run boundaries."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"lag(n={n!r}): need a positive integer")
    return AccMetric("lag", col, n)


def convolution(col: str, kernel) -> AccMetric:
    """A 1-D convolution / FIR filter: the dot product of ``kernel`` with the last ``len(kernel)``
    values of ``col`` (oldest·``kernel[0]`` … current·``kernel[-1]``) — NULL until the group has
    ``len(kernel)`` rows; NULL inputs count as 0. Output is a DOUBLE."""
    kernel = tuple(kernel)
    if not kernel:
        raise ValueError("convolution(kernel=...): the kernel must be non-empty")
    return AccMetric("conv", col, init=kernel)


def scan(fn, init, dtype: str = "double") -> AccMetric:
    """A **custom fold**: ``fn(state, row) -> (new_state, output)`` applied per row in ``.along``
    order; ``init`` is the per-group starting state, ``row`` a ``{column: value}`` dict. The state is
    JSON-persisted between runs; ``fn`` is pickled to the executors, so keep it self-contained.
    ``dtype`` is the output's Spark type string."""
    return AccMetric("scan", None, None, fn=fn, init=init, dtype=dtype)
