"""Session fixtures: one local Delta-enabled SparkSession, one throwaway database per test.

Everything runs offline on ``local[2]`` — the engine is pure OSS Spark + Delta, so CI never needs a
Databricks workspace. Delta's per-commit log work dominates test time; the tiny partition counts below
keep each commit cheap.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def _jdk_major(home: str) -> int:
    try:
        with open(os.path.join(home, "release")) as fh:
            for line in fh:
                if line.startswith("JAVA_VERSION="):
                    return int(line.split('"')[1].split(".")[0])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _prefer_arrow_capable_jdk() -> None:
    """Spark 3.5 bundles Arrow Java 12, which cannot allocate direct buffers on JDK 18+ — every
    ``applyInPandas`` (the accumulate fold) would fail. When the ambient JDK is too new and a JDK 17
    is installed, point the Spark JVM at it. Managed runtimes (Databricks) ship a compatible pairing."""
    current = os.environ.get("JAVA_HOME", "")
    if current and 0 < _jdk_major(current) <= 17:
        return
    for cand in sorted(glob.glob("/usr/lib/jvm/java-17-*")):
        if os.path.exists(os.path.join(cand, "bin", "java")):
            os.environ["JAVA_HOME"] = cand
            return


@pytest.fixture(scope="session")
def spark():
    _prefer_arrow_capable_jdk()
    # Spark's Python workers must run the same interpreter (and site-packages) as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    warehouse = tempfile.mkdtemp(prefix="trickle_spark_wh_")
    builder = (
        SparkSession.builder.appName("trickle-spark-tests")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.databricks.delta.snapshotPartitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # Arrow (applyInPandas) on JDK 17/21 needs reflective access the module system closed off;
        # Databricks/spark-submit set these themselves — this is purely for the bare local JVM.
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
    shutil.rmtree(warehouse, ignore_errors=True)


_counter = 0


@pytest.fixture
def db(spark):
    """A fresh database per test, dropped afterwards, so table names never collide across tests."""
    global _counter
    _counter += 1
    name = f"t{_counter}"
    spark.sql(f"CREATE DATABASE {name}")
    yield name
    spark.sql(f"DROP DATABASE {name} CASCADE")
