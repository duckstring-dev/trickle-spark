"""Phase-1 behaviour: watermarks, windows, the MERGE apply, and the fallback rungs.

These are ports of the reference scenarios in duckstring's ``tests/test_trickle.py`` translated from
epoch mechanics to version-window mechanics: replay/idempotence becomes "re-run → empty windows →
skip", and the anti-cascade property is asserted on the *output's own CDF* — a comprehensive run must
not make downstream windows any bigger than the real change.
"""

from __future__ import annotations

from delta.tables import DeltaTable

import trickle_spark as ts


def make_source(spark, name, rows, schema="id INT, v STRING"):
    df = spark.createDataFrame(rows, schema)
    df.write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(name)


def latest_version(spark, name) -> int:
    return DeltaTable.forName(spark, name).history(1).collect()[0].version


def cdf_at(spark, name, version):
    return (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", version)
        .option("endingVersion", version)
        .table(name)
        .collect()
    )


def rows_of(spark, name):
    return sorted((r.id, r.v) for r in spark.table(name).collect())


def identity(src):
    """The identity plan: output mirrors the source. `full` reads pinned; `delta` passes the window."""

    def full(ctx):
        return ctx.new(src)

    def delta(ctx):
        d = ctx.delta(src)
        return None if d.is_full else d.zset

    return {"sources": [src], "pk": ("id",), "full": full, "delta": delta}


def test_bootstrap_creates_output_with_cdf_and_watermarks(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a"), (2, "b")])
    res = ts.run(spark, out, **identity(src))
    assert res.status == "bootstrap"
    assert rows_of(spark, out) == [(1, "a"), (2, "b")]
    props = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {out}").collect()}
    assert props.get("delta.enableChangeDataFeed") == "true"
    wm = ts.read_watermarks(spark, out)
    assert wm is not None and wm[src].version == 0 and wm[src].table_id


def test_rerun_with_no_change_skips_without_committing(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a")])
    ts.run(spark, out, **identity(src))
    before = latest_version(spark, out)
    res = ts.run(spark, out, **identity(src))
    assert res.status == "skipped"
    assert latest_version(spark, out) == before  # no commit at all


def test_incremental_insert_update_delete_propagate(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a"), (2, "b"), (3, "c")])
    ts.run(spark, out, **identity(src))

    spark.sql(f"INSERT INTO {src} VALUES (4, 'd')")
    spark.sql(f"UPDATE {src} SET v = 'B' WHERE id = 2")
    spark.sql(f"DELETE FROM {src} WHERE id = 3")
    res = ts.run(spark, out, **identity(src))
    assert res.status == "incremental"
    assert rows_of(spark, out) == [(1, "a"), (2, "B"), (4, "d")]

    # the output's own change feed — the next consumer's window — covers only the touched keys
    changed = cdf_at(spark, out, latest_version(spark, out))
    assert sorted({r.id for r in changed}) == [2, 3, 4]


def test_multi_commit_source_window_consolidates(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a")])
    ts.run(spark, out, **identity(src))
    # a→b→c across three commits nets to one update; an insert-then-delete nets to nothing
    spark.sql(f"UPDATE {src} SET v = 'b' WHERE id = 1")
    spark.sql(f"UPDATE {src} SET v = 'c' WHERE id = 1")
    spark.sql(f"INSERT INTO {src} VALUES (9, 'ghost')")
    spark.sql(f"DELETE FROM {src} WHERE id = 9")
    res = ts.run(spark, out, **identity(src))
    assert res.status == "incremental"
    assert rows_of(spark, out) == [(1, "c")]
    changed = cdf_at(spark, out, latest_version(spark, out))
    assert {r.id for r in changed} == {1}  # the ghost never reaches the output's feed


def test_comprehensive_diff_keeps_downstream_window_small(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a"), (2, "b"), (3, "c")])
    plan = identity(src)
    ts.run(spark, out, **plan)
    spark.sql(f"UPDATE {src} SET v = 'B' WHERE id = 2")
    res = ts.run(spark, out, sources=plan["sources"], pk=plan["pk"], full=plan["full"])  # no delta fn
    assert res.status == "comprehensive"
    assert rows_of(spark, out) == [(1, "a"), (2, "B"), (3, "c")]
    changed = cdf_at(spark, out, latest_version(spark, out))
    assert {r.id for r in changed} == {2}  # recomputed everything, emitted only the real change


def test_recreated_source_is_full_and_run_recovers(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a"), (2, "b")])
    ts.run(spark, out, **identity(src))
    spark.sql(f"DROP TABLE {src}")
    make_source(spark, src, [(1, "a"), (5, "e")])  # fresh table id, version counter reset
    res = ts.run(spark, out, **identity(src))
    assert res.status == "comprehensive"  # identity's delta fn sees is_full and declines
    assert rows_of(spark, out) == [(1, "a"), (5, "e")]


def test_p_threshold_reads_full_past_the_change_fraction(spark, db):
    src = f"{db}.src"
    make_source(spark, src, [(i, "x") for i in range(10)])
    from trickle_spark.tables import pin_table

    last = pin_table(spark, src)
    spark.sql(f"UPDATE {src} SET v = 'y' WHERE id < 8")  # 8/10 rows churn
    pin = pin_table(spark, src)
    assert ts.delta_of(spark, src, pin, last, p=0.5).is_full
    assert not ts.delta_of(spark, src, pin, last, p=1.0).is_full  # 1.0 disables the check
    assert not ts.delta_of(spark, src, pin, last).is_full


def test_empty_output_delta_still_advances_the_watermark(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "keep"), (2, "drop")])

    def full(ctx):
        return ctx.new(src).where("v = 'keep'")

    def delta(ctx):
        d = ctx.delta(src)
        return None if d.is_full else d.zset.where("v = 'keep' OR v = 'kept'")

    ts.run(spark, out, sources=[src], pk=("id",), full=full, delta=delta)
    spark.sql(f"UPDATE {src} SET v = 'dropped' WHERE id = 2")  # churn that never reaches the output
    res = ts.run(spark, out, sources=[src], pk=("id",), full=full, delta=delta)
    assert res.status == "incremental"
    assert rows_of(spark, out) == [(1, "keep")]
    # the empty commit advanced the watermark: the next run skips instead of re-reading the window
    res = ts.run(spark, out, sources=[src], pk=("id",), full=full, delta=delta)
    assert res.status == "skipped"
    # and the empty commit emitted no CDF rows — downstream consumers cascade skips, not work
    assert cdf_at(spark, out, latest_version(spark, out)) == []


def test_two_sources_one_changed(spark, db):
    a, b, out = f"{db}.a", f"{db}.b", f"{db}.out"
    make_source(spark, a, [(1, "a1")])
    make_source(spark, b, [(1, "b1")], schema="id INT, w STRING")

    def full(ctx):
        return ctx.new(a).join(ctx.new(b), "id")

    ts.run(spark, out, sources=[a, b], pk=("id",), full=full)
    spark.sql(f"UPDATE {a} SET v = 'a2' WHERE id = 1")
    res = ts.run(spark, out, sources=[a, b], pk=("id",), full=full)
    assert res.status == "comprehensive"
    assert sorted((r.id, r.v, r.w) for r in spark.table(out).collect()) == [(1, "a2", "b1")]
    ctx_wm = ts.read_watermarks(spark, out)
    assert ctx_wm[a].version == 1 and ctx_wm[b].version == 0  # both recorded at their pins


def test_foreign_commits_on_output_do_not_lose_watermarks(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a")])
    ts.run(spark, out, **identity(src))
    spark.sql(f"ALTER TABLE {out} SET TBLPROPERTIES ('foo' = 'bar')")  # a non-trickle commit on top
    res = ts.run(spark, out, **identity(src))
    assert res.status == "skipped"  # watermarks recovered from the older trickle commit


def test_existing_table_without_watermarks_goes_comprehensive(spark, db):
    src, out = f"{db}.src", f"{db}.out"
    make_source(spark, src, [(1, "a"), (2, "b")])
    # someone else made the output (no trickle metadata, stale content)
    spark.createDataFrame([(1, "stale")], "id INT, v STRING").write.format("delta").saveAsTable(out)
    res = ts.run(spark, out, **identity(src))
    assert res.status == "comprehensive"
    assert rows_of(spark, out) == [(1, "a"), (2, "b")]
    assert ts.read_watermarks(spark, out) is not None  # adopted: from here on it's watermark-managed
