"""Aggregation behaviour: O(δ) folds, retraction rescans, the Chan/Pébay moment merge, and the
state-companion commit order — asserted against Spark's own aggregates over the current input as the
numerical oracle, and against the output's CDF for the only-affected-groups property."""

from __future__ import annotations

import math

import pytest
from helpers import cdf_at, latest_version, make_source, table_rows

import trickle_spark as ts
from trickle_spark import BuildError, agg
from trickle_spark.aggregate import state_table_for


def lines(spark, db, rows=None):
    make_source(
        spark,
        f"{db}.lines",
        rows or [(1, "A", 2, 10.0), (2, "A", 1, 30.0), (3, "B", 5, 3.0), (4, "B", 1, 7.0)],
        schema="line_id INT, product STRING, qty INT, price DOUBLE",
    )


def revenue_plan(db):
    return (
        ts.source(f"{db}.lines", p=1.0).alias("li")
        .mutate(amount="li.qty * li.price")
        .aggregate(by="product", orders=agg.count(), total=agg.sum("amount"), avg_qty=agg.mean("qty"))
    )


def test_sum_count_mean_fold_incrementally(spark, db):
    lines(spark, db)
    out = f"{db}.revenue"
    plan = revenue_plan(db)
    res = plan.merge_into(spark, out)  # pk defaults to the group key
    assert res.status == "bootstrap"
    assert table_rows(spark, out, "product", "orders", "total", "avg_qty") == [
        ("A", 2, 50.0, 1.5), ("B", 2, 22.0, 3.0)]

    spark.sql(f"INSERT INTO {db}.lines VALUES (5, 'A', 4, 5.0)")
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert table_rows(spark, out, "product", "orders", "total", "avg_qty") == [
        ("A", 3, 70.0, 7.0 / 3.0), ("B", 2, 22.0, 3.0)]
    # only the touched group reaches the output's change feed
    assert {r.product for r in cdf_at(spark, out, latest_version(spark, out))} == {"A"}

    spark.sql(f"DELETE FROM {db}.lines WHERE line_id = 2")  # a retraction folds out
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert table_rows(spark, out, "product", "orders", "total") == [("A", 2, 40.0), ("B", 2, 22.0)]


def test_emptied_group_is_retracted(spark, db):
    lines(spark, db)
    out = f"{db}.revenue"
    plan = revenue_plan(db)
    plan.merge_into(spark, out)
    spark.sql(f"DELETE FROM {db}.lines WHERE product = 'B'")
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert [r[0] for r in table_rows(spark, out, "product")] == ["A"]
    changed = cdf_at(spark, out, latest_version(spark, out))
    assert {(r.product, r._change_type) for r in changed} == {("B", "delete")}


def test_min_max_extend_and_rescan(spark, db):
    lines(spark, db)
    out = f"{db}.extremes"
    plan = (ts.source(f"{db}.lines", p=1.0)
            .aggregate(by="product", lo=agg.min("price"), hi=agg.max("price")))
    plan.merge_into(spark, out)
    assert table_rows(spark, out, "product", "lo", "hi") == [("A", 10.0, 30.0), ("B", 3.0, 7.0)]

    spark.sql(f"INSERT INTO {db}.lines VALUES (6, 'A', 1, 50.0)")  # insert extends in place
    plan.merge_into(spark, out)
    assert table_rows(spark, out, "product", "hi") == [("A", 50.0), ("B", 7.0)]

    spark.sql(f"DELETE FROM {db}.lines WHERE line_id = 6")  # the supporting max is retracted → rescan
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert table_rows(spark, out, "product", "lo", "hi") == [("A", 10.0, 30.0), ("B", 3.0, 7.0)]
    assert {r.product for r in cdf_at(spark, out, latest_version(spark, out))} == {"A"}


def test_var_stddev_match_spark_after_mixed_windows(spark, db):
    make_source(spark, f"{db}.m", [(i, "g", float(i * i % 17)) for i in range(1, 12)],
                schema="id INT, g STRING, x DOUBLE")
    out = f"{db}.stats"
    plan = (ts.source(f"{db}.m", p=1.0)
            .aggregate(by="g", v=agg.var("x"), sd=agg.stddev("x"), vp=agg.var("x", "pop")))
    plan.merge_into(spark, out)
    # several windows of inserts, updates and deletes — the merge-in/merge-out path, repeatedly
    spark.sql(f"INSERT INTO {db}.m VALUES (20, 'g', 40.0), (21, 'g', 2.5)")
    plan.merge_into(spark, out)
    spark.sql(f"UPDATE {db}.m SET x = x + 3.25 WHERE id % 3 = 0")
    plan.merge_into(spark, out)
    spark.sql(f"DELETE FROM {db}.m WHERE id IN (2, 5, 20)")
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    exp = spark.sql(f"SELECT var_samp(x) v, stddev_samp(x) sd, var_pop(x) vp FROM {db}.m").collect()[0]
    got = spark.table(out).collect()[0]
    assert math.isclose(got.v, exp.v, rel_tol=1e-9)
    assert math.isclose(got.sd, exp.sd, rel_tol=1e-9)
    assert math.isclose(got.vp, exp.vp, rel_tol=1e-9)


def test_weighted_product_arg_and_bool_families(spark, db):
    make_source(spark, f"{db}.w",
                [(1, "g", 2.0, 1.0, True), (2, "g", 3.0, 3.0, True), (3, "g", -4.0, 2.0, False)],
                schema="id INT, g STRING, x DOUBLE, w DOUBLE, ok BOOLEAN")
    out = f"{db}.waggs"
    plan = (ts.source(f"{db}.w", p=1.0)
            .aggregate(by="g",
                       wavg=agg.weighted_average("x", "w"), wtot=agg.weight_total("w"),
                       prod=agg.product("x"), best=agg.argmax("id", "x"), all_ok=agg.bool_and("ok")))
    plan.merge_into(spark, out)
    r = spark.table(out).collect()[0]
    assert math.isclose(r.wavg, (2.0 + 9.0 - 8.0) / 6.0)
    assert r.wtot == 6.0 and r.best == 2 and r.all_ok is False
    assert math.isclose(r.prod, -24.0)

    spark.sql(f"DELETE FROM {db}.w WHERE id = 3")  # retraction: argmax unaffected, bool_and rescans
    spark.sql(f"INSERT INTO {db}.w VALUES (4, 'g', 5.0, 4.0, true)")
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    r = spark.table(out).collect()[0]
    assert math.isclose(r.wavg, (2.0 + 9.0 + 20.0) / 8.0)
    assert r.wtot == 8.0 and r.best == 4 and r.all_ok is True
    assert math.isclose(r.prod, 30.0)


def test_aggregate_over_a_join(spark, db):
    make_source(spark, f"{db}.orders", [(1, "A", 2), (2, "A", 1), (3, "B", 5)],
                schema="order_id INT, product_id STRING, qty INT")
    make_source(spark, f"{db}.cat", [("A", 10.0), ("B", 3.0)], schema="product_id STRING, price DOUBLE")
    out = f"{db}.rev"
    plan = (ts.source(f"{db}.orders", p=1.0).alias("o")
            .join(ts.source(f"{db}.cat", p=1.0).alias("c"), on="product_id")
            .mutate(amount="o.qty * c.price")
            .aggregate(by="product_id", revenue=agg.sum("amount")))
    plan.merge_into(spark, out)
    assert table_rows(spark, out, "product_id", "revenue") == [("A", 30.0), ("B", 15.0)]

    spark.sql(f"UPDATE {db}.cat SET price = 20.0 WHERE product_id = 'A'")  # dim change → group A only
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert table_rows(spark, out, "product_id", "revenue") == [("A", 60.0), ("B", 15.0)]
    assert {r.product_id for r in cdf_at(spark, out, latest_version(spark, out))} == {"A"}


def test_comprehensive_rebuild_keeps_state_and_output_consistent(spark, db):
    lines(spark, db)
    out = f"{db}.revenue"
    plan = revenue_plan(db)
    plan.merge_into(spark, out)
    spark.sql(f"UPDATE {db}.lines SET qty = qty + 1")  # 100% churn on a p=1.0 source stays incremental
    res = plan.merge_into(spark, out, ivm=False)  # force the comprehensive rung instead
    assert res.status == "comprehensive"
    assert table_rows(spark, out, "product", "orders", "total") == [("A", 2, 90.0), ("B", 2, 32.0)]
    # and the run after the rebuild folds incrementally again, off the rebuilt state
    spark.sql(f"INSERT INTO {db}.lines VALUES (7, 'B', 2, 1.0)")
    res = plan.merge_into(spark, out)
    assert res.status == "incremental"
    assert table_rows(spark, out, "product", "total") == [("A", 90.0), ("B", 34.0)]


def test_state_companion_records_pins_and_metric_change_rebuilds(spark, db):
    lines(spark, db)
    out = f"{db}.revenue"
    plan = revenue_plan(db)
    plan.merge_into(spark, out)
    state = state_table_for(out)
    assert ts.read_watermarks(spark, state) == ts.read_watermarks(spark, out)

    # a different metric set behind the same output columns → the state schema no longer matches →
    # comprehensive rebuild (a change to the *output* schema is a new table, not an evolution)
    plan2 = (ts.source(f"{db}.lines", p=1.0).alias("li")
             .mutate(amount="li.qty * li.price")
             .aggregate(by="product", orders=agg.count(), total=agg.sum("amount"), avg_qty=agg.max("qty")))
    spark.sql(f"INSERT INTO {db}.lines VALUES (8, 'A', 1, 100.0)")
    res = plan2.merge_into(spark, out)
    assert res.status == "comprehensive"
    assert table_rows(spark, out, "product", "orders", "total", "avg_qty") == [
        ("A", 3, 150.0, 2), ("B", 2, 22.0, 5)]


def test_aggregate_build_errors(spark, db):
    lines(spark, db)
    with pytest.raises(BuildError, match="group key"):
        ts.source(f"{db}.lines").aggregate(total=agg.sum("qty"))
    with pytest.raises(BuildError, match="metric"):
        ts.source(f"{db}.lines").aggregate(by="product", total="sum(qty)")
    with pytest.raises(BuildError, match="follow"):
        ts.source(f"{db}.lines").aggregate(by="product", n=agg.count()).filter("n > 1")
    with pytest.raises(BuildError, match="operand"):
        ts.source(f"{db}.lines").join(
            ts.source(f"{db}.lines").alias("l2").aggregate(by="product", n=agg.count()), on="product")
