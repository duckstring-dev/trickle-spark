"""Ordered-scan behaviour: tail resume from carried state, past-edit refolds, and the merge-diff that
keeps the output's change feed to the rows whose running values actually changed."""

from __future__ import annotations

import math

import pytest
from helpers import cdf_at, latest_version, make_source, table_rows

import trickle_spark as ts
from trickle_spark import BuildError, acc
from trickle_spark.accumulate import state_table_for


def events(spark, db):
    make_source(
        spark,
        f"{db}.events",
        [(1, "g1", 1, 10.0), (2, "g1", 2, 20.0), (3, "g2", 1, 5.0)],
        schema="event_id INT, grp STRING, t INT, x DOUBLE",
    )


def scan_plan(db):
    return (
        ts.source(f"{db}.events", p=1.0)
        .along("t")
        .accumulate(by="grp", run_total=acc.sum("x"), seen=acc.count(), peak=acc.max("x"))
    )


def rows_of(spark, out):
    return table_rows(spark, out, "event_id", "run_total", "seen", "peak")


def test_tail_append_resumes_from_carried_state(spark, db):
    events(spark, db)
    out = f"{db}.scored"
    plan = scan_plan(db)
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "bootstrap"
    assert rows_of(spark, out) == [(1, 10.0, 1, 10.0), (2, 30.0, 2, 20.0), (3, 5.0, 1, 5.0)]

    spark.sql(f"INSERT INTO {db}.events VALUES (4, 'g1', 3, 5.0), (5, 'g2', 2, 1.0)")
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "incremental"
    assert rows_of(spark, out) == [
        (1, 10.0, 1, 10.0), (2, 30.0, 2, 20.0), (3, 5.0, 1, 5.0), (4, 35.0, 3, 20.0), (5, 6.0, 2, 5.0)]
    # a tail append emits only the new rows — the running values of old rows are untouched
    assert sorted({r.event_id for r in cdf_at(spark, out, latest_version(spark, out))}) == [4, 5]


def test_past_edit_refolds_the_group_and_diffs(spark, db):
    events(spark, db)
    out = f"{db}.scored"
    plan = scan_plan(db)
    plan.merge_into(spark, out, pk=("event_id",))
    spark.sql(f"INSERT INTO {db}.events VALUES (4, 'g1', 3, 5.0)")
    plan.merge_into(spark, out, pk=("event_id",))

    spark.sql(f"UPDATE {db}.events SET x = 100.0 WHERE event_id = 2")  # an edit below g1's high-water
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "incremental"
    assert rows_of(spark, out) == [
        (1, 10.0, 1, 10.0), (2, 110.0, 2, 100.0), (3, 5.0, 1, 5.0), (4, 115.0, 3, 100.0)]
    # the refold emitted the whole group, but the diff kept row 1 (unchanged) out of the change feed
    assert sorted({r.event_id for r in cdf_at(spark, out, latest_version(spark, out))}) == [2, 4]


def test_past_delete_refolds(spark, db):
    events(spark, db)
    out = f"{db}.scored"
    plan = scan_plan(db)
    plan.merge_into(spark, out, pk=("event_id",))
    spark.sql(f"DELETE FROM {db}.events WHERE event_id = 1")
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "incremental"
    assert rows_of(spark, out) == [(2, 20.0, 1, 20.0), (3, 5.0, 1, 5.0)]
    # and a later tail append still resumes correctly from the refolded state
    spark.sql(f"INSERT INTO {db}.events VALUES (6, 'g1', 5, 1.0)")
    plan.merge_into(spark, out, pk=("event_id",))
    assert rows_of(spark, out) == [(2, 20.0, 1, 20.0), (3, 5.0, 1, 5.0), (6, 21.0, 2, 20.0)]


def test_emptied_group_state_is_dropped(spark, db):
    events(spark, db)
    out = f"{db}.scored"
    plan = scan_plan(db)
    plan.merge_into(spark, out, pk=("event_id",))
    spark.sql(f"DELETE FROM {db}.events WHERE grp = 'g2'")
    plan.merge_into(spark, out, pk=("event_id",))
    assert [r[0] for r in rows_of(spark, out)] == [1, 2]
    state = spark.table(state_table_for(out))
    assert [r.grp for r in state.collect()] == ["g1"]  # g2's carried state must not survive
    # a revived g2 starts a fresh scan, not a resume from the dead state
    spark.sql(f"INSERT INTO {db}.events VALUES (7, 'g2', 9, 2.0)")
    plan.merge_into(spark, out, pk=("event_id",))
    assert (7, 2.0, 1, 2.0) in rows_of(spark, out)


def test_ema_lag_first_and_custom_scan(spark, db):
    make_source(spark, f"{db}.ticks", [(1, "s", 1, 10.0), (2, "s", 2, 20.0), (3, "s", 3, 30.0)],
                schema="id INT, sym STRING, t INT, px DOUBLE")
    out = f"{db}.ticked"

    def volatility(state, row):  # a custom fold: running |Δpx|
        prev = state.get("px")
        state["px"] = row["px"]
        return state, (None if prev is None else abs(row["px"] - prev))

    plan = (ts.source(f"{db}.ticks", p=1.0)
            .along("t")
            .accumulate(by="sym",
                        smooth=acc.ema("px", 0.5), prior=acc.prev("px"),
                        opening=acc.first("px"), jump=acc.scan(volatility, {})))
    plan.merge_into(spark, out, pk=("id",))
    got = {r.id: r for r in spark.table(out).collect()}
    assert got[1].smooth == 10.0 and math.isclose(got[2].smooth, 15.0) and math.isclose(got[3].smooth, 22.5)
    assert got[1].prior is None and got[2].prior == 10.0 and got[3].prior == 20.0
    assert all(got[i].opening == 10.0 for i in (1, 2, 3))
    assert got[1].jump is None and got[2].jump == 10.0 and got[3].jump == 10.0

    spark.sql(f"INSERT INTO {db}.ticks VALUES (4, 's', 4, 2.5)")  # every carried state crosses the run
    plan.merge_into(spark, out, pk=("id",))
    r4 = spark.table(out).where("id = 4").collect()[0]
    assert math.isclose(r4.smooth, 12.5) and r4.prior == 30.0 and r4.opening == 10.0 and r4.jump == 27.5


def test_comprehensive_matches_incremental(spark, db):
    events(spark, db)
    out_a, out_b = f"{db}.inc", f"{db}.comp"
    plan = scan_plan(db)
    plan.merge_into(spark, out_a, pk=("event_id",))
    plan.merge_into(spark, out_b, pk=("event_id",))
    spark.sql(f"INSERT INTO {db}.events VALUES (4, 'g1', 3, 5.0)")
    spark.sql(f"UPDATE {db}.events SET x = 7.0 WHERE event_id = 3")
    res_a = plan.merge_into(spark, out_a, pk=("event_id",))
    res_b = plan.merge_into(spark, out_b, pk=("event_id",), ivm=False)
    assert res_a.status == "incremental" and res_b.status == "comprehensive"
    assert rows_of(spark, out_a) == rows_of(spark, out_b)


def test_ungrouped_scan_folds_the_whole_table(spark, db):
    events(spark, db)
    out = f"{db}.scored"
    # no by= — one fold over the whole table; event_id is the globally monotonic axis
    plan = (ts.source(f"{db}.events", p=1.0)
            .along("event_id")
            .accumulate(run_total=acc.sum("x"), seen=acc.count()))
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "bootstrap"
    assert table_rows(spark, out, "event_id", "run_total", "seen") == [
        (1, 10.0, 1), (2, 30.0, 2), (3, 35.0, 3)]

    spark.sql(f"INSERT INTO {db}.events VALUES (4, 'g2', 3, 2.0)")  # a tail append resumes
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "event_id", "run_total", "seen") == [
        (1, 10.0, 1), (2, 30.0, 2), (3, 35.0, 3), (4, 37.0, 4)]
    assert {r.event_id for r in cdf_at(spark, out, latest_version(spark, out))} == {4}

    spark.sql(f"DELETE FROM {db}.events WHERE event_id = 2")  # a past change re-folds everything after it
    res = plan.merge_into(spark, out, pk=("event_id",))
    assert res.status == "incremental"
    assert table_rows(spark, out, "event_id", "run_total", "seen") == [
        (1, 10.0, 1), (3, 15.0, 2), (4, 17.0, 3)]


def test_accumulate_build_errors(spark, db):
    events(spark, db)
    src = ts.source(f"{db}.events")
    with pytest.raises(BuildError, match="order axis"):
        src.accumulate(by="grp", total=acc.sum("x"))
    with pytest.raises(BuildError, match="acc\\.\\*"):
        ts.source(f"{db}.events").along("t").accumulate(by="grp", total="sum(x)")
    with pytest.raises(BuildError, match="follow"):
        ts.source(f"{db}.events").along("t").accumulate(by="grp", total=acc.sum("x")).filter("total > 1")
    with pytest.raises(BuildError, match="row key"):
        ts.source(f"{db}.events").along("t").accumulate(by="grp", total=acc.sum("x")).merge_into(
            spark, f"{db}.x")