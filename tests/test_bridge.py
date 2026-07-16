"""The bridge primitives: apply_changes (the tag-idempotent Z-set sink) and table_changes (the
windowed reader for a foreign consumer that keeps its own pin)."""

from __future__ import annotations

import pytest
from helpers import cdf_at, latest_version, make_source, table_rows

import trickle_spark as ts
from trickle_spark import D_COL
from trickle_spark.tables import read_tag


def zset(spark, rows, schema="id INT, v STRING", weights=None):
    df = spark.createDataFrame(rows, schema)
    if weights is None:
        return ts.as_zset(df, 1)
    pairs = [(*r, w) for r, w in zip(rows, weights, strict=True)]
    return spark.createDataFrame(pairs, f"{schema}, {D_COL} LONG")


def test_apply_changes_bootstraps_then_merges(spark, db):
    land = f"{db}.landed"
    assert ts.apply_changes(spark, land, zset(spark, [(1, "a"), (2, "b")]), pk=("id",), tag="f1")
    assert table_rows(spark, land, "id", "v") == [(1, "a"), (2, "b")]
    props = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {land}").collect()}
    assert props.get("delta.enableChangeDataFeed") == "true"  # a landing table is a native source

    # an update travels as -old +new; a delete as a bare -1 — the full-row Z-set contract
    change = zset(spark, [(2, "b"), (2, "B"), (1, "a")], weights=[-1, 1, -1])
    assert ts.apply_changes(spark, land, change, pk=("id",), tag="f2")
    assert table_rows(spark, land, "id", "v") == [(2, "B")]
    kinds = {(r.id, r._change_type) for r in cdf_at(spark, land, latest_version(spark, land))}
    assert kinds == {(1, "delete"), (2, "update_preimage"), (2, "update_postimage")}


def test_apply_changes_same_tag_is_a_noop(spark, db):
    land = f"{db}.landed"
    ts.apply_changes(spark, land, zset(spark, [(1, "a")]), pk=("id",), tag="f1")
    v = latest_version(spark, land)
    change = zset(spark, [(1, "a"), (1, "A")], weights=[-1, 1])
    assert ts.apply_changes(spark, land, change, pk=("id",), tag="f2")
    assert not ts.apply_changes(spark, land, change, pk=("id",), tag="f2")  # the replay lands nothing
    assert latest_version(spark, land) == v + 1
    assert read_tag(spark, land) == "f2"


def test_apply_changes_full_state_diffs_against_the_landing(spark, db):
    land = f"{db}.landed"
    ts.apply_changes(spark, land, zset(spark, [(1, "a"), (2, "b"), (3, "c")]), pk=("id",), tag="f1")
    # a foreign coverage-miss hands over the complete state: 1 unchanged, 2 changed, 3 gone, 4 new
    state = zset(spark, [(1, "a"), (2, "B"), (4, "d")])
    assert ts.apply_changes(spark, land, state, pk=("id",), tag="f2", full=True)
    assert table_rows(spark, land, "id", "v") == [(1, "a"), (2, "B"), (4, "d")]
    # the diff kept the unchanged row out of the change feed — downstream windows stay honest
    assert {r.id for r in cdf_at(spark, land, latest_version(spark, land))} == {2, 3, 4}


def test_apply_changes_duplicate_landing_key_raises(spark, db):
    with pytest.raises(ValueError, match="not unique"):
        ts.apply_changes(spark, f"{db}.landed", zset(spark, [(1, "a"), (1, "b")]), pk=("id",))


def test_table_changes_windows_by_the_callers_pin(spark, db):
    src = f"{db}.src"
    make_source(spark, src, [(1, "a"), (2, "b")])
    d, pin = ts.table_changes(spark, src, None)
    assert d.is_full  # no pin yet — the whole current state
    assert sorted((r.id, r.v) for r in d.zset.drop(D_COL).collect()) == [(1, "a"), (2, "b")]

    spark.sql(f"UPDATE {src} SET v = 'B' WHERE id = 2")
    spark.sql(f"INSERT INTO {src} VALUES (3, 'c')")
    d, pin2 = ts.table_changes(spark, src, pin)
    assert not d.is_full
    changed = sorted((r.id, r.v, r[D_COL]) for r in d.zset.collect())
    assert changed == [(2, "B", 1), (2, "b", -1), (3, "c", 1)]

    d, _ = ts.table_changes(spark, src, pin2)
    assert not d.is_full and d.is_empty()  # caught up — a free stable operand