"""trickle-spark: DBSP-style incremental data transforms on Spark + Delta Lake.

A standalone sibling of Duckstring's Trickle engine (the reference implementation), re-based on what
the lakehouse already provides: Delta Change Data Feed as the Z-set changelog, time travel as the old
state, one MERGE commit per run carrying its own watermarks. No epochs, no control plane, no resident
service — see ``docs/design.md`` for the design.
"""

from . import (
    acc,  # noqa: F401 - the scan-metric namespace (from trickle_spark import acc)
    agg,  # noqa: F401 - the aggregate-metric namespace (from trickle_spark import agg)
)
from .builder import Builder, BuildError, materialize, source
from .changes import delta_of
from .run import RunContext, RunResult, run
from .tables import Pin, read_watermarks
from .zset import D_COL, SYSTEM_PREFIX, Delta, as_zset, consolidate, diff

__all__ = [
    "BuildError",
    "Builder",
    "D_COL",
    "SYSTEM_PREFIX",
    "Delta",
    "Pin",
    "RunContext",
    "RunResult",
    "as_zset",
    "consolidate",
    "delta_of",
    "diff",
    "materialize",
    "read_watermarks",
    "run",
    "source",
]
