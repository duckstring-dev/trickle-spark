"""Append-only outputs (.append_to): insert-only commits, the conflict semantics (retractions and
changed-past images), the droplog companion, and the benign idempotent skips that keep comprehensive
re-derivations from spuriously failing."""

from __future__ import annotations

import pytest
from helpers import cdf_at, latest_version, make_source, table_rows

import trickle_spark as ts
from trickle_spark import AppendConflict, BuildError, acc, agg
from trickle_spark.append import droplog_for
from trickle_spark.zset import D_COL


def orders(spark, db):
    make_source(
        spark,
        f"{db}.orders",
        [(1, "widget", 2), (2, "gadget", 1), (3, "widget", 0)],
        schema="order_id INT, product STRING, qty INT",
    )


def clean_plan(db):
    return ts.source(f"{db}.orders", p=1.0).alias("o").filter("o.qty > 0")


def test_append_bootstrap_incremental_and_skip(spark, db):
    orders(spark, db)
    out = f"{db}.clean"
    plan = clean_plan(db)
    res = plan.append_to(spark, out, pk=("order_id",))
    assert res.status == "bootstrap"
    assert table_rows(spark, out, "order_id", "qty") == [(1, 2), (2, 1)]

    spark.sql(f"INSERT INTO {db}.orders VALUES (4, 'widget', 5)")
    res = plan.append_to(spark, out, pk=("order_id",))
    assert res.status == "incremental" and res.changed
    assert table_rows(spark, out, "order_id", "qty") == [(1, 2), (2, 1), (4, 5)]
    # the commit is a pure insert — an append-only table's CDF carries no updates or deletes
    assert {r._change_type for r in cdf_at(spark, out, latest_version(spark, out))} == {"insert"}

    res = plan.append_to(spark, out, pk=("order_id",))
    assert res.status == "skipped" and not res.changed


def test_append_conflict_raises_by_default(spark, db):
    orders(spark, db)
    out = f"{db}.clean"
    plan = clean_plan(db)
    plan.append_to(spark, out, pk=("order_id",))

    spark.sql(f"UPDATE {db}.orders SET qty = 9 WHERE order_id = 1")  # a change to the past
    with pytest.raises(AppendConflict, match="not append-safe"):
        plan.append_to(spark, out, pk=("order_id",))
    # nothing was written — history intact, and the watermark didn't advance (the run will re-raise)
    assert table_rows(spark, out, "order_id", "qty") == [(1, 2), (2, 1)]


def test_append_drops_conflicts_and_logs_them(spark, db):
    orders(spark, db)
    out = f"{db}.clean"
    plan = clean_plan(db)
    plan.append_to(spark, out, pk=("order_id",))

    # one past change (a retraction + a changed image) and one genuinely new row, in one window
    spark.sql(f"UPDATE {db}.orders SET qty = 9 WHERE order_id = 1")
    spark.sql(f"INSERT INTO {db}.orders VALUES (5, 'gizmo', 7)")
    res = plan.append_to(spark, out, pk=("order_id",), fail_on_conflict=False)
    assert res.status == "incremental" and res.changed
    # history wins: order 1 keeps its original image; the new row landed
    assert table_rows(spark, out, "order_id", "qty") == [(1, 2), (2, 1), (5, 7)]
    # the droplog holds the retracted old image (-1) and the rejected new image (+1)
    dropped = sorted((r.order_id, r.qty, r[D_COL]) for r in spark.table(droplog_for(out)).collect())
    assert dropped == [(1, 2, -1), (1, 9, 1)]

    res = plan.append_to(spark, out, pk=("order_id",), fail_on_conflict=False)
    assert res.status == "skipped"


def test_append_within_run_duplicate_pk_always_raises(spark, db):
    make_source(spark, f"{db}.raw", [(1, "a"), (1, "b")], schema="k INT, v STRING")
    with pytest.raises(AppendConflict, match="not unique"):
        ts.source(f"{db}.raw", p=1.0).append_to(spark, f"{db}.hist", pk=("k",), fail_on_conflict=False)


def test_append_comprehensive_rederivation_skips_identical_rows(spark, db):
    orders(spark, db)
    out = f"{db}.clean"
    plan = clean_plan(db)
    plan.append_to(spark, out, pk=("order_id",))

    spark.sql(f"INSERT INTO {db}.orders VALUES (6, 'widget', 3)")
    res = plan.append_to(spark, out, pk=("order_id",), ivm=False)  # comprehensive: everything re-derived
    assert res.status == "comprehensive" and res.changed
    # re-derived identical rows skipped benignly — only the new row hit the table
    assert {r.order_id for r in cdf_at(spark, out, latest_version(spark, out))} == {6}
    assert table_rows(spark, out, "order_id", "qty") == [(1, 2), (2, 1), (6, 3)]


def test_append_empty_output_delta_heartbeats_the_watermark(spark, db):
    orders(spark, db)
    out = f"{db}.clean"
    plan = clean_plan(db)
    plan.append_to(spark, out, pk=("order_id",))

    spark.sql(f"UPDATE {db}.orders SET product = 'brick' WHERE order_id = 3")  # stays filtered out
    res = plan.append_to(spark, out, pk=("order_id",))
    assert res.status == "incremental" and not res.changed
    res = plan.append_to(spark, out, pk=("order_id",))  # the heartbeat advanced the watermark
    assert res.status == "skipped"


def test_append_terminal_build_errors(spark, db):
    orders(spark, db)
    with pytest.raises(BuildError, match="aggregate"):
        (ts.source(f"{db}.orders").aggregate(by="product", n=agg.count())
         .append_to(spark, f"{db}.x"))
    with pytest.raises(BuildError, match="accumulate"):
        (ts.source(f"{db}.orders").along("order_id").accumulate(by="product", n=acc.count())
         .append_to(spark, f"{db}.x"))
