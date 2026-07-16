"""``run`` — one incremental maintenance step for one output table.

A pure library call with no control plane: it discovers its own watermarks from the output's commit
history, pins the sources, plans windows, computes, and lands one commit. Idempotent and self-contained
per invocation — callable from a notebook, a Workflows task, or a plain script. The standing assumption
is **one writer per output table** (Delta's optimistic concurrency is the backstop, not the mechanism).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pyspark.sql import DataFrame, SparkSession

from . import duckstring as _dk
from .apply import apply_zset, current, write_bootstrap
from .changes import delta_of
from .tables import Pin, commit_metadata_json, pin_table, read_watermarks, table_exists
from .zset import Delta, diff


def pin_of(spark: SparkSession, source: str) -> Pin:
    """Pin a source for this run — a Delta table's version, or a duckstring source's published epoch."""
    if _dk.is_duckstring_ref(source):
        return _dk.pin_source(source)
    return pin_table(spark, source)


@dataclass
class RunContext:
    """What a compute function sees: pinned states and watermark windows per source.

    - ``new(source)`` — the source **as of its pin** (reads are pinned, never "latest", so concurrent
      producer commits land in the next window).
    - ``old(source)`` — the source as of its watermark: time travel **is** the old state; nothing is
      reconstructed.
    - ``delta(source)`` — the consolidated Z-set window between the two (see :func:`~.changes.delta_of`
      for the full-read fallback rungs).
    """

    spark: SparkSession
    pins: dict[str, Pin]
    last: dict[str, Pin] | None
    p: dict[str, float]
    _deltas: dict[str, Delta] = field(default_factory=dict)

    def new(self, source: str) -> DataFrame:
        if _dk.is_duckstring_ref(source):
            return _dk.current_state(self.spark, source, self.pins[source])
        from .tables import read_as_of

        return read_as_of(self.spark, source, self.pins[source].version)

    def old(self, source: str) -> DataFrame:
        if _dk.is_duckstring_ref(source):
            d = self.delta(source)
            if d.is_full:
                raise ValueError(f"{source}: no usable watermark — there is no old state (check Delta.is_full first)")
            return _dk.rewind(self.new(source), d.zset)  # current ⊎ (−δ): no time travel needed
        from .tables import read_as_of

        entry = (self.last or {}).get(source)
        if entry is None or entry.table_id != self.pins[source].table_id:
            raise ValueError(f"{source}: no usable watermark — there is no old state (check Delta.is_full first)")
        return read_as_of(self.spark, source, entry.version)

    def delta(self, source: str) -> Delta:
        if source not in self._deltas:
            if _dk.is_duckstring_ref(source):
                self._deltas[source] = _dk.delta_of_ref(
                    self.spark, source, self.pins[source], (self.last or {}).get(source), p=self.p.get(source)
                )
            else:
                self._deltas[source] = delta_of(
                    self.spark, source, self.pins[source], (self.last or {}).get(source), p=self.p.get(source)
                )
        return self._deltas[source]


@dataclass(frozen=True)
class RunResult:
    status: str  # "bootstrap" | "incremental" | "comprehensive" | "skipped"
    pins: dict[str, Pin]
    changed: bool = True  # False when the run wrote no data rows (a skip, or an empty-delta heartbeat)


def run(
    spark: SparkSession,
    output: str,
    *,
    sources: list[str],
    pk: tuple[str, ...],
    full: Callable[[RunContext], DataFrame],
    delta: Callable[[RunContext], DataFrame | None] | None = None,
    p: float | dict[str, float] | None = None,
    tag: str | None = None,
) -> RunResult:
    """One maintenance step for ``output``.

    - ``full(ctx)`` (required): the complete output from the sources' pinned states.
    - ``delta(ctx)`` (optional): the output's Z-set change from the sources' windows; return ``None``
      to force a comprehensive run (e.g. when a needed source reads ``is_full``). Without it every
      non-skip run is comprehensive — still useful, still watermark-managed.
    - ``p``: change-fraction threshold(s) past which a source reads as full (float for all sources, or
      per-source dict).

    The decision ladder: all windows empty → **skip** (no commit, nothing downstream fires); no table →
    **bootstrap** (create, CDF on); no usable watermark or no/declined ``delta`` → **comprehensive**
    (recompute, then *diff against the current output* and MERGE only real changes, so downstream
    windows stay honest); otherwise → **incremental** (apply ``delta``'s Z-set). Every non-skip path
    lands exactly one commit carrying the pinned watermarks.

    ``tag`` stamps the run's commit with an opaque caller token (e.g. a Duckstring run's epoch ``f``)
    — :func:`~.changes.changes_at_tag` can then recover exactly this run's change, content-addressed.
    """
    if not sources:
        raise ValueError("run() needs at least one source")
    p_map = dict(p) if isinstance(p, dict) else {s: p for s in sources} if p is not None else {}

    exists = table_exists(spark, output)
    last = read_watermarks(spark, output) if exists else None
    pins = {s: pin_of(spark, s) for s in sources}

    if exists and last is not None and all(last.get(s) == pins[s] for s in sources):
        return RunResult(status="skipped", pins=pins, changed=False)

    ctx = RunContext(spark=spark, pins=pins, last=last, p=p_map)

    if not exists:
        write_bootstrap(spark, output, full(ctx), commit_metadata_json("bootstrap", pins, tag=tag))
        return RunResult(status="bootstrap", pins=pins)

    zset = None
    status = "comprehensive"
    if last is not None and delta is not None:
        zset = delta(ctx)
        status = "incremental" if zset is not None else "comprehensive"
    if zset is None:
        zset = diff(full(ctx), current(spark, output))
    action = apply_zset(spark, output, zset, pk, commit_metadata_json(status, pins, tag=tag))
    return RunResult(status=status, pins=pins, changed=action == "merged")
