"""Unit tests for Serving & Scale Analysis benchmarking module."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.serving_benchmark import (
    calculate_cost_and_qps,
    measure_ann_index_memory,
    measure_feature_store_memory,
    run_serving_benchmarks,
)


def test_measure_ann_index_memory():
    """Verify ANN memory calculation returns positive, expected byte sizes."""
    mem_mind = measure_ann_index_memory(n_articles=65238, dim=384, use_faiss=True)
    assert mem_mind["n_articles"] == 65238
    assert mem_mind["embedding_dim"] == 384
    assert mem_mind["raw_matrix_mb"] > 0
    assert mem_mind["total_index_ram_mb"] > mem_mind["raw_matrix_mb"]
    assert mem_mind["projected_10x_ram_gb"] > 0
    # 65238 * 384 * 4 bytes = 100,199,424 bytes ~ 95.56 MB
    assert 90.0 < mem_mind["raw_matrix_mb"] < 105.0


def test_measure_feature_store_memory():
    """Verify feature store memory metrics return valid sizes."""
    mem_mind = measure_feature_store_memory("mind", scale="small")
    assert mem_mind["n_articles"] > 0
    assert mem_mind["article_store_disk_mb"] > 0
    assert mem_mind["duckdb_working_ram_mb"] > 0
    assert mem_mind["total_store_disk_mb"] > 0
    assert mem_mind["projected_10x_store_disk_gb"] > 0


def test_calculate_cost_and_qps():
    """Verify throughput and cost per 1,000 queries arithmetic."""
    dummy_lat = {
        50: {
            "total_mean_ms": 5.0,
            "total_p50_ms": 4.5,
            "total_p90_ms": 6.5,
            "total_p95_ms": 7.5,
            "total_p99_ms": 12.0,
        },
        100: {
            "total_mean_ms": 8.0,
            "total_p50_ms": 7.0,
            "total_p90_ms": 10.0,
            "total_p95_ms": 12.0,
            "total_p99_ms": 18.0,
        },
    }
    cost_res = calculate_cost_and_qps(
        latency_results=dummy_lat,
        target_sla_ms=100.0,
        hourly_rate_usd=0.34,
        vcpus=8,
    )
    assert "K=50" in cost_res
    assert "K=100" in cost_res
    assert cost_res["K=50"]["meets_sla"] is True
    assert cost_res["K=100"]["meets_sla"] is True
    assert cost_res["K=50"]["instance_qps"] > cost_res["K=100"]["instance_qps"]
    assert cost_res["K=50"]["cost_per_1k_queries_usd"] < cost_res["K=100"]["cost_per_1k_queries_usd"]
    assert cost_res["K=50"]["cost_per_1k_queries_usd"] > 0.0


def test_e2e_serving_benchmarks():
    """Verify run_serving_benchmarks runs end-to-end and outputs structured DataFrames."""
    df_serv, df_lat = run_serving_benchmarks(dataset="mind", n_iterations=15)
    assert len(df_serv) == 3  # K=50, 100, 200
    assert len(df_lat) == 3
    assert "candidate_k" in df_serv.columns
    assert "p99_latency_ms" in df_serv.columns
    assert "cost_per_1k_queries_usd" in df_serv.columns

    # Verify percentiles ordering
    for row in df_lat.iter_rows(named=True):
        assert row["total_p50_ms"] <= row["total_p90_ms"] <= row["total_p95_ms"] <= row["total_p99_ms"]
        assert row["total_p99_ms"] < 100.0  # Must comfortably satisfy SLA
