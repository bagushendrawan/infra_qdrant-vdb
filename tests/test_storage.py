from src.storage.config import Settings
from src.storage.service import Store


def test_storage(tmp_path):
    settings = Settings(data_path=str(tmp_path))
    store = Store(settings)
    store.execute("create", "test", {})
    dense = [1.0] + [0.0] * 1023
    store.execute(
        "upsert",
        "test",
        {
            "records": [
                {
                    "id": 1,
                    "dense_vec": dense,
                    "lexical_weights": {"1": 1.0},
                    "colbert_vecs": [dense],
                    "metadata": {"group": "a"},
                },
                {"id": 2, "dense_vec": [0.0, 1.0] + [0.0] * 1022},
            ]
        },
    )
    for mode in ["dense", "sparse", "colbert", "hybrid", "rerank"]:
        result = store.execute(
            "query",
            "test",
            {
                "mode": mode,
                "dense_vec": dense,
                "lexical_weights": {"1": 1.0},
                "colbert_vecs": [dense],
                "limit": 1,
                "candidate_limit": 10,
                "modalities": ["dense", "sparse"],
            },
        )
        assert result["records"][0]["id"] == 1
    store.close()
    store = Store(settings)
    assert len(store.execute("get", "test", {"ids": [1, 2]})["records"]) == 2
    store.execute("upsert", "test", {"records": [{"id": 1, "dense_vec": dense}]})
    record = store.execute("get", "test", {"ids": [1], "with_vectors": True})["records"][0]
    assert "sparse" not in record["vector"]
    store.execute("delete", "test", {"ids": [1, 2]})
    assert store.execute("get", "test", {"ids": [1, 2]})["records"] == []
    store.close()


def test_different_winners_and_filters(tmp_path):
    store = Store(Settings(data_path=str(tmp_path)))
    a, b = [1.0] + [0.0] * 1023, [0.0, 1.0] + [0.0] * 1022
    try:
        store.execute("create", "rank", {})
        store.execute(
            "upsert",
            "rank",
            {
                "records": [
                    {"id": 1, "dense_vec": a, "lexical_weights": {"1": 0.1}, "colbert_vecs": [b]},
                    {"id": 2, "dense_vec": b, "lexical_weights": {"1": 2.0}, "colbert_vecs": [b]},
                    {"id": 3, "dense_vec": b, "lexical_weights": {"1": 0.2}, "colbert_vecs": [a]},
                ]
            },
        )
        query = dict(
            dense_vec=a,
            lexical_weights={"1": 1.0},
            colbert_vecs=[a],
            limit=1,
            candidate_limit=10,
            modalities=["dense", "sparse", "colbert"],
        )
        for mode, winner in [("dense", 1), ("sparse", 2), ("colbert", 3), ("rerank", 3)]:
            assert (
                store.execute("query", "rank", dict(query, mode=mode))["records"][0]["id"] == winner
            )
        result = store.execute(
            "query", "rank", dict(query, mode="hybrid", filter={"must": [{"has_id": [2]}]})
        )
        assert [p["id"] for p in result["records"]] == [2]
    finally:
        store.close()
