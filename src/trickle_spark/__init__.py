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
from .append import AppendConflict, run_append
from .bridge import apply_changes, table_changes
from .builder import Builder, BuildError, materialize, materialize_append, source
from .changes import changes_at_tag, delta_of
from .duckstring import duckstring_source
from .run import RunContext, RunResult, run
from .tables import Pin, pin_table, read_tag, read_watermarks
from .zset import D_COL, SYSTEM_PREFIX, Delta, as_zset, consolidate, diff

__all__ = [
    "AppendConflict",
    "BuildError",
    "Builder",
    "D_COL",
    "SYSTEM_PREFIX",
    "Delta",
    "Pin",
    "RunContext",
    "RunResult",
    "apply_changes",
    "as_zset",
    "changes_at_tag",
    "consolidate",
    "delta_of",
    "diff",
    "duckstring_source",
    "materialize",
    "materialize_append",
    "pin_table",
    "read_tag",
    "read_watermarks",
    "run",
    "run_append",
    "source",
    "table_changes",
]
