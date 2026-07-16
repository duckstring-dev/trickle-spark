"""Builder behaviour: the affected-key recompute across join types, the output pipeline, and the
fallback/strategy rungs — ported from the reference suite's scenarios (duckstring ``tests/``, builder
cases), asserted on both the output rows and the output's **own change feed** (the next consumer's
window must cover exactly the touched keys)."""

from __future__ import annotations

import pytest
from helpers import cdf_at, latest_version, make_source, table_rows

import trickle_spark as ts
from trickle_spark import BuildError


def orders_catalog(spark, db):
    """The reference demo shape: an orders fact + a catalog dim joined on product_id."""
    make_source(spark, f"{db}.orders", [(1, "A", 2), (2, "A", 1), (3, "B", 5)],
                schema="order_id INT, product_id STRING, qty INT")
    make_source(spark, f"{db}.catalog", [("A", 10.0), ("B", 3.0)],
                schema="product_id STRING, price DOUBLE")


def priced_plan(db):
    return (
        ts.source(f"{db}.orders", p=1.0).alias("o")
        .join(ts.source(f"{db}.catalog", p=1.0).alias("c"), on="product_id", how="left")
        .select("o.order_id, o.product_id, o.qty, c.price")
    )


def test_inner_join_dim_update_touches_only_affected_keys(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.priced"
    plan = priced_plan(db)
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "bootstrap"
    assert table_rows(spark, out, "order_id", "price") == [(1, 10.0), (2, 10.0), (3, 3.0)]

    spark.sql(f"UPDATE {db}.catalog SET price = 12.0 WHERE product_id = 'A'")
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "order_id", "price") == [(1, 12.0), (2, 12.0), (3, 3.0)]
    # only the A-keyed orders appear in the output's change feed; order 3 (product B) is untouched
    assert sorted({r.order_id for r in cdf_at(spark, out, latest_version(spark, out))}) == [1, 2]


def test_fact_and_dim_change_in_one_window(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.priced"
    plan = priced_plan(db)
    plan.merge_into(spark, out, pk=("order_id",))
    spark.sql(f"INSERT INTO {db}.orders VALUES (4, 'B', 7)")
    spark.sql(f"UPDATE {db}.catalog SET price = 11.0 WHERE product_id = 'A'")
    spark.sql(f"DELETE FROM {db}.orders WHERE order_id = 2")
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "order_id", "qty", "price") == [(1, 2, 11.0), (3, 5, 3.0), (4, 7, 3.0)]


def test_left_join_incomparables_convert_both_ways(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.priced"
    plan = priced_plan(db)
    spark.sql(f"INSERT INTO {db}.orders VALUES (9, 'Z', 1)")  # no catalog match → NULL-padded
    plan.merge_into(spark, out, pk=("order_id",))
    assert (9, None) in table_rows(spark, out, "order_id", "price")

    spark.sql(f"INSERT INTO {db}.catalog VALUES ('Z', 99.0)")  # the incomparable gains its match
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "incremental"
    assert (9, 99.0) in table_rows(spark, out, "order_id", "price")
    assert {r.order_id for r in cdf_at(spark, out, latest_version(spark, out))} == {9}

    spark.sql(f"DELETE FROM {db}.catalog WHERE product_id = 'Z'")  # and loses it again
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "incremental"
    assert (9, None) in table_rows(spark, out, "order_id", "price")


def test_bushy_join_composes(spark, db):
    make_source(spark, f"{db}.a", [(1, "a1")], schema="k INT, va STRING")
    make_source(spark, f"{db}.b", [(1, "b1")], schema="k INT, vb STRING")
    make_source(spark, f"{db}.c", [(1, "c1")], schema="k INT, vc STRING")
    make_source(spark, f"{db}.d", [(1, "d1")], schema="k INT, vd STRING")
    out = f"{db}.wide"
    left = ts.source(f"{db}.a", p=1.0).alias("a").join(ts.source(f"{db}.b", p=1.0).alias("b"), on="k")
    right = ts.source(f"{db}.c", p=1.0).alias("c").join(ts.source(f"{db}.d", p=1.0).alias("d"), on="k")
    plan = left.join(right, on={"a.k": "c.k"}).select("a.k, a.va, b.vb, c.vc, d.vd")
    plan.merge_into(spark, out, pk=("k",))
    assert table_rows(spark, out, "k", "va", "vb", "vc", "vd") == [(1, "a1", "b1", "c1", "d1")]

    spark.sql(f"UPDATE {db}.d SET vd = 'D!' WHERE k = 1")
    res = plan.merge_into(spark, out, pk=("k",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "k", "vd") == [(1, "D!")]


def test_semi_and_anti_joins_maintain_membership(spark, db):
    orders_catalog(spark, db)
    known, unknown = f"{db}.known", f"{db}.unknown"
    spark.sql(f"INSERT INTO {db}.orders VALUES (9, 'Z', 1)")
    semi = ts.source(f"{db}.orders", p=1.0).alias("o").join(ts.source(f"{db}.catalog", p=1.0).alias("c"),
                                                     on="product_id", how="semi")
    anti = ts.source(f"{db}.orders", p=1.0).alias("o").join(ts.source(f"{db}.catalog", p=1.0).alias("c"),
                                                     on="product_id", how="anti")
    semi.merge_into(spark, known, pk=("order_id",))
    anti.merge_into(spark, unknown, pk=("order_id",))
    assert [r[0] for r in table_rows(spark, known, "order_id")] == [1, 2, 3]
    assert [r[0] for r in table_rows(spark, unknown, "order_id")] == [9]

    spark.sql(f"INSERT INTO {db}.catalog VALUES ('Z', 5.0)")  # order 9 flips from anti to semi
    assert semi.merge_into(spark, known, pk=("order_id",)).status == "incremental"
    assert anti.merge_into(spark, unknown, pk=("order_id",)).status == "incremental"
    assert [r[0] for r in table_rows(spark, known, "order_id")] == [1, 2, 3, 9]
    assert table_rows(spark, unknown, "order_id") == []


def test_full_outer_pads_both_sides(spark, db):
    make_source(spark, f"{db}.l", [(1, "l1"), (2, "l2")], schema="k INT, lv STRING")
    make_source(spark, f"{db}.r", [(2, "r2"), (3, "r3")], schema="k INT, rv STRING")
    out = f"{db}.both"
    plan = (ts.source(f"{db}.l", p=1.0).alias("l").join(ts.source(f"{db}.r", p=1.0).alias("r"), on="k", how="full")
            .mutate(key="coalesce(l.k, r.k)")
            .select("key, l.lv, r.rv"))
    plan.merge_into(spark, out, pk=("key",))
    assert table_rows(spark, out, "key", "lv", "rv") == [(1, "l1", None), (2, "l2", "r2"), (3, None, "r3")]

    spark.sql(f"INSERT INTO {db}.l VALUES (3, 'l3')")  # the right-side incomparable gains its match
    res = plan.merge_into(spark, out, pk=("key",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "key", "lv", "rv") == [(1, "l1", None), (2, "l2", "r2"), (3, "l3", "r3")]
    assert {r.key for r in cdf_at(spark, out, latest_version(spark, out))} == {3}


def test_pipeline_filter_after_mutate_sees_the_mutated_column(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.big"
    plan = (priced_plan_base(db)
            .mutate(amount="o.qty * c.price")
            .filter("amount > 10")
            .select("o.order_id, amount"))
    plan.merge_into(spark, out, pk=("order_id",))
    assert table_rows(spark, out, "order_id", "amount") == [(1, 20.0), (3, 15.0)]  # order 2: 1*10 filtered

    spark.sql(f"UPDATE {db}.orders SET qty = 3 WHERE order_id = 2")  # 3*10 → now passes the filter
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "order_id", "amount") == [(1, 20.0), (2, 30.0), (3, 15.0)]


def priced_plan_base(db):
    return ts.source(f"{db}.orders", p=1.0).alias("o").join(ts.source(f"{db}.catalog", p=1.0).alias("c"), on="product_id")


def test_star_output_dedups_join_keys_and_rejects_other_collisions(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.star"
    plan = priced_plan_base(db)  # no .select — the bare * output
    plan.merge_into(spark, out, pk=("order_id",))
    cols = spark.table(out).columns
    assert cols.count("product_id") == 1  # the equi-join key appears once
    assert set(cols) == {"order_id", "product_id", "qty", "price"}

    make_source(spark, f"{db}.l2", [(1, "x")], schema="k INT, v STRING")
    make_source(spark, f"{db}.r2", [(1, "y")], schema="k INT, v STRING")
    bad = ts.source(f"{db}.l2", p=1.0).alias("l").join(ts.source(f"{db}.r2", p=1.0).alias("r"), on="k")
    with pytest.raises(BuildError, match="ambiguous"):
        bad.merge_into(spark, f"{db}.bad", pk=("k",))


def test_p_threshold_falls_back_comprehensively_and_stays_correct(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.priced"
    plan = (  # default p=0.3 — the whole-dim churn below must trip it
        ts.source(f"{db}.orders").alias("o")
        .join(ts.source(f"{db}.catalog").alias("c"), on="product_id", how="left")
        .select("o.order_id, o.product_id, o.qty, c.price")
    )
    plan.merge_into(spark, out, pk=("order_id",))
    spark.sql(f"UPDATE {db}.catalog SET price = price + 1")  # 100% of the dim churns → default p=0.3 trips
    res = plan.merge_into(spark, out, pk=("order_id",))
    assert res.status == "comprehensive"
    assert table_rows(spark, out, "order_id", "price") == [(1, 11.0), (2, 11.0), (3, 4.0)]


def test_strategy_escapes_ivm_and_key_filter(spark, db):
    orders_catalog(spark, db)
    a, b = f"{db}.via_ivm_off", f"{db}.via_kf_off"
    plan = priced_plan(db)
    plan.merge_into(spark, a, pk=("order_id",), ivm=False)
    plan.merge_into(spark, b, pk=("order_id",), key_filter=False)
    spark.sql(f"UPDATE {db}.catalog SET price = 7.0 WHERE product_id = 'B'")
    res_a = plan.merge_into(spark, a, pk=("order_id",), ivm=False)
    res_b = plan.merge_into(spark, b, pk=("order_id",), key_filter=False)
    assert res_a.status == "comprehensive" and res_b.status == "incremental"
    expected = [(1, 10.0), (2, 10.0), (3, 7.0)]
    assert table_rows(spark, a, "order_id", "price") == expected
    assert table_rows(spark, b, "order_id", "price") == expected
    # both keep the downstream window honest: only the touched key in the change feed
    assert {r.order_id for r in cdf_at(spark, a, latest_version(spark, a))} == {3}
    assert {r.order_id for r in cdf_at(spark, b, latest_version(spark, b))} == {3}


def test_sql_escape_is_comprehensive_but_output_stays_incremental(spark, db):
    orders_catalog(spark, db)
    out = f"{db}.by_product"
    plan = (priced_plan_base(db).alias("t")
            .sql("SELECT product_id, sum(qty * price) AS revenue FROM t GROUP BY product_id"))
    res = plan.merge_into(spark, out, pk=("product_id",))
    assert res.status == "bootstrap"
    assert table_rows(spark, out, "product_id", "revenue") == [("A", 30.0), ("B", 15.0)]

    spark.sql(f"UPDATE {db}.orders SET qty = 6 WHERE order_id = 3")  # only product B's revenue moves
    res = plan.merge_into(spark, out, pk=("product_id",))
    assert res.status == "comprehensive"
    assert table_rows(spark, out, "product_id", "revenue") == [("A", 30.0), ("B", 18.0)]
    assert {r.product_id for r in cdf_at(spark, out, latest_version(spark, out))} == {"B"}


def test_dict_on_joins_differently_named_keys(spark, db):
    make_source(spark, f"{db}.fact", [(1, 10), (2, 20)], schema="fid INT, val INT")
    make_source(spark, f"{db}.dim", [(1, "one"), (2, "two")], schema="did INT, name STRING")
    out = f"{db}.named"
    plan = (ts.source(f"{db}.fact", p=1.0).alias("f")
            .join(ts.source(f"{db}.dim", p=1.0).alias("d"), on={"fid": "did"})
            .select("f.fid, f.val, d.name"))
    plan.merge_into(spark, out, pk=("fid",))
    spark.sql(f"UPDATE {db}.dim SET name = 'TWO' WHERE did = 2")
    res = plan.merge_into(spark, out, pk=("fid",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "fid", "name") == [(1, "one"), (2, "TWO")]


def test_build_errors(spark, db):
    orders_catalog(spark, db)
    src = ts.source(f"{db}.orders", p=1.0)
    with pytest.raises(BuildError, match="one of"):
        src.join(ts.source(f"{db}.catalog", p=1.0), on="product_id", how="cross")
    with pytest.raises(BuildError, match="operand"):
        ts.source(f"{db}.orders", p=1.0).join(ts.source(f"{db}.catalog", p=1.0).filter("price > 1"), on="product_id")
    with pytest.raises(BuildError, match="pass the output key"):
        ts.materialize(spark, f"{db}.x", ts.source(f"{db}.orders", p=1.0), pk=())
    with pytest.raises(BuildError, match="reserved"):
        ts.source(f"{db}.orders", p=1.0).mutate(_trickle_bad="1")
    with pytest.raises(BuildError, match="alias"):
        ts.source(f"{db}.orders", p=1.0).sql("SELECT 1")
    with pytest.raises(BuildError, match="not found"):
        priced_plan_base(db).join(ts.source(f"{db}.catalog", p=1.0).alias("c2"), on="nope").merge_into(
            spark, f"{db}.y", pk=("order_id",))
    with pytest.raises(BuildError, match="missing the PK"):
        (ts.source(f"{db}.orders", p=1.0).alias("o").select("o.qty").merge_into(spark, f"{db}.z", pk=("order_id",)))
    with pytest.raises(BuildError, match="duplicate source alias"):
        (ts.source(f"{db}.orders", p=1.0).alias("o")
         .join(ts.source(f"{db}.catalog", p=1.0).alias("o"), on="product_id")
         .merge_into(spark, f"{db}.w", pk=("order_id",)))


def test_schema_and_count(spark, db):
    orders_catalog(spark, db)
    plan = priced_plan(db)
    sch = plan.schema(spark)
    assert sch == {"order_id": "int", "product_id": "string", "qty": "int", "price": "double"}
    assert plan.count(spark) == 3
