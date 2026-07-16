"""Aggregate metric specs for the builder's ``.aggregate(by=…, name=agg.…)`` operator.

Each metric is a small typed spec, not a SQL string, because the incremental engine needs to know its
*kind* — which accumulators it maintains and how it updates from a delta (see ``aggregate.py``):

- **Distributive / algebraic, pure O(δ)** — ``count()``, ``sum(col)``, ``mean(col)``, ``var(col)`` /
  ``stddev(col)`` (the centred second moment ``M2 = Σ(x − x̄)²`` maintained by the parallel Chan/Pébay
  merge-in/merge-out — retractable *and* well-conditioned, never ``Σx² − (Σx)²/n``), the weighted
  family ``weight_total(w)`` / ``weighted_sum(x, w)`` / ``weighted_average(x, w)`` (pure sums), and
  ``product(col)`` (retractable log-sum-exp: sign count + ``Σ log|x|`` → a DOUBLE, not bit-exact for
  large integer products).
- **Two-variable co-moments, pure O(δ)** — ``covariance(x, y)`` / ``pearson_correlation(x, y)`` /
  ``ols_slope(x, y)`` / ``ols_intercept(x, y)``: the paired accumulator ``(n, Σx, Σy, M2x, M2y, Cxy)``
  over rows where **both** columns are non-NULL (pairwise deletion), maintained well-conditioned by
  the same Pébay merge-in/merge-out as ``var`` — the centred sums are never ``Σxy − ΣxΣy/n``.
- **Extend-or-rescan** — ``min``/``max``, ``argmin``/``argmax``, ``bool_and``/``bool_or``/
  ``bit_and``/``bit_or``: an insert extends the stored value in place (O(δ)); a group with any
  retraction **rescans** its current membership (the supporting row may be gone).
- **Order-dependent** — ``reduce(fn, init)``: a custom fold in ``.along(...)`` order collapsed to one
  value per group (the reducing counterpart of :func:`trickle_spark.acc.scan`). Requires ``.along``,
  can't mix with the order-independent metrics in one ``.aggregate()``; maintained by the accumulate
  machinery's tail-resume / past-refold split (see ``accumulate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    """One output aggregate: its ``kind``, the input column(s) it reads, and — for ``var``/``stddev``/
    ``covariance`` — whether it's a ``sample`` or ``pop``ulation statistic. ``fn``/``init``/``dtype``
    carry the reducer for :func:`reduce`."""

    kind: str
    col: str | None = None
    how: str | None = None
    col2: str | None = None
    fn: object = None
    init: object = None
    dtype: str | None = None


def count() -> Metric:
    """Count of rows in the group (``count(*)``)."""
    return Metric("count")


def sum(col: str) -> Metric:  # noqa: A001 - deliberate SQL-style name on the agg namespace
    """Running sum of ``col`` (NULLs ignored; an all-NULL group is NULL, per SQL ``sum``)."""
    return Metric("sum", col)


def mean(col: str) -> Metric:
    """Mean of ``col`` over its non-NULL values — algebraic, maintained as ``sum(col)/count(col)``."""
    return Metric("mean", col)


def min(col: str) -> Metric:  # noqa: A001 - deliberate SQL-style name on the agg namespace
    """Minimum of ``col`` (NULLs ignored). A retraction of the supporting row rescans the group."""
    return Metric("min", col)


def max(col: str) -> Metric:  # noqa: A001 - deliberate SQL-style name on the agg namespace
    """Maximum of ``col`` (NULLs ignored). A retraction of the supporting row rescans the group."""
    return Metric("max", col)


def var(col: str, how: str = "sample") -> Metric:
    """Variance of ``col`` over its non-NULL values. ``how`` ∈ ``"sample"`` (default) / ``"pop"``."""
    return Metric("var", col, _check_how(how))


def stddev(col: str, how: str = "sample") -> Metric:
    """Standard deviation of ``col`` over its non-NULL values. ``how`` ∈ ``"sample"`` / ``"pop"``."""
    return Metric("stddev", col, _check_how(how))


def product(col: str) -> Metric:
    """Product of ``col`` over its non-NULL values (any 0 → 0; all-NULL → NULL). Maintained
    retractably via sign/zero counts + ``Σ log|x|`` — the result is a DOUBLE."""
    return Metric("product", col)


def weight_total(w: str) -> Metric:
    """Sum of the weights ``Σw`` over rows where ``w`` is non-NULL."""
    return Metric("weight_total", w)


def weighted_sum(x: str, w: str) -> Metric:
    """Weighted sum ``Σ(w·x)`` over rows where both ``x`` and ``w`` are non-NULL."""
    return Metric("weighted_sum", x, col2=w)


def weighted_average(x: str, w: str) -> Metric:
    """Weighted mean ``Σ(w·x) / Σw`` over rows where both are non-NULL (NULL if ``Σw`` = 0)."""
    return Metric("weighted_average", x, col2=w)


# ─── two-variable co-moments (paired; maintained by the parallel Pébay merge) ─────
#
# All four read the paired accumulator ``(n, Σx, Σy, M2x, M2y, Cxy)`` over rows where *both* columns
# are non-NULL (pairwise deletion). The centred sums are maintained well-conditioned (never
# ``Σxy − ΣxΣy/n``); see ``aggregate.py``.


def covariance(x: str, y: str, how: str = "sample") -> Metric:
    """Covariance of ``x`` and ``y`` — ``Cxy / (n-1)`` (sample, default; NULL for n<2) or ``Cxy / n`` (pop)."""
    return Metric("covariance", x, _check_how(how), col2=y)


def pearson_correlation(x: str, y: str) -> Metric:
    """Pearson correlation ``Cxy / sqrt(M2x · M2y)`` (NULL when n<2 or either spread is 0)."""
    return Metric("pearson_correlation", x, col2=y)


def ols_slope(x: str, y: str) -> Metric:
    """Ordinary-least-squares slope of ``y`` on ``x`` — ``Cxy / M2x`` (NULL when ``x`` has no spread)."""
    return Metric("ols_slope", x, col2=y)


def ols_intercept(x: str, y: str) -> Metric:
    """Ordinary-least-squares intercept of ``y`` on ``x`` — ``ȳ − slope·x̄`` (NULL when ``x`` has no spread)."""
    return Metric("ols_intercept", x, col2=y)


def reduce(fn, init, *, dtype: str = "double") -> Metric:  # noqa: A001 - the reduce primitive
    """A **custom order-dependent reduction** — one value per group, the final result of folding the
    group's rows in ``.along`` order: ``fn(state, row) -> (new_state, output)``, ``init`` the per-group
    start, ``row`` a ``{column: value}`` dict; the group's value is the last ``output``. The
    order-dependent counterpart of the order-independent ``agg.*`` reductions, and the reducing
    counterpart of :func:`trickle_spark.acc.scan`.

    Requires ``.along(...)``; can't share an ``.aggregate()`` with order-independent metrics.
    Retraction-aware: a change anywhere in a group re-folds it over current membership; a tail-only
    append resumes from the group's carried fold-state. ``fn`` is pickled to the executors and the
    state is JSON-persisted between runs — keep both self-contained. ``dtype`` is the output's Spark
    type string."""
    return Metric("reduce", fn=fn, init=init, dtype=dtype)


def argmin(arg: str, by: str) -> Metric:
    """The ``arg`` value at the row where ``by`` is minimal (ties arbitrary). Rescans on retraction."""
    return Metric("argmin", arg, col2=by)


def argmax(arg: str, by: str) -> Metric:
    """The ``arg`` value at the row where ``by`` is maximal (ties arbitrary). Rescans on retraction."""
    return Metric("argmax", arg, col2=by)


def bool_and(col: str) -> Metric:
    """Logical AND over ``col`` (NULLs ignored). A retraction rescans the group."""
    return Metric("bool_and", col)


def bool_or(col: str) -> Metric:
    """Logical OR over ``col`` (NULLs ignored). A retraction rescans the group."""
    return Metric("bool_or", col)


def bit_and(col: str) -> Metric:
    """Bitwise AND over an integer ``col`` (NULLs ignored). A retraction rescans the group."""
    return Metric("bit_and", col)


def bit_or(col: str) -> Metric:
    """Bitwise OR over an integer ``col`` (NULLs ignored). A retraction rescans the group."""
    return Metric("bit_or", col)


def _check_how(how: str) -> str:
    h = how.lower()
    if h in ("pop", "population"):
        return "pop"
    if h == "sample":
        return "sample"
    raise ValueError(f"agg how={how!r}: one of 'sample' / 'pop'")
