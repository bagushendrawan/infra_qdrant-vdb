from qdrant_client import QdrantClient
from qdrant_client import models as m

from src.storage.exceptions import StorageError


class Store:
    def __init__(self, settings):
        self.client = QdrantClient(path=settings.data_path)

    def close(self):
        self.client.close()

    def execute(self, operation, name, payload):
        c = self.client
        if operation == "list":
            return {"collections": [v.name for v in c.get_collections().collections]}
        exists = c.collection_exists(name)
        if operation == "create":
            if exists:
                raise StorageError(409, "Collection exists")
            c.create_collection(
                name,
                vectors_config={
                    "dense": m.VectorParams(size=1024, distance=m.Distance.COSINE),
                    "colbert": m.VectorParams(
                        size=1024,
                        distance=m.Distance.COSINE,
                        multivector_config=m.MultiVectorConfig(
                            comparator=m.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                },
                sparse_vectors_config={"sparse": m.SparseVectorParams()},
            )
            return {"name": name}
        if not exists:
            raise StorageError(404, "Collection not found")
        if operation == "info":
            return {"name": name, **c.get_collection(name).model_dump(mode="json")}
        if operation == "drop":
            c.delete_collection(name)
            return {"name": name}
        if operation == "upsert":
            points = []
            for r in payload["records"]:
                vectors = {"dense": r["dense_vec"]}
                if r.get("lexical_weights") is not None:
                    vectors["sparse"] = self.sparse(r["lexical_weights"])
                if r.get("colbert_vecs") is not None:
                    vectors["colbert"] = r["colbert_vecs"]
                points.append(
                    m.PointStruct(
                        id=r["id"],
                        vector=vectors,
                        payload={"document": r.get("document"), "metadata": r.get("metadata")},
                    )
                )
            c.upsert(name, points, wait=True)
            return {"ids": [p.id for p in points]}
        if operation == "delete":
            c.delete(name, m.PointIdsList(points=payload["ids"]), wait=True)
            return {"ids": payload["ids"]}
        if operation == "get":
            points = c.retrieve(
                name,
                payload["ids"],
                with_payload=True,
                with_vectors=payload.get("with_vectors", False),
            )
        else:
            q = payload
            vectors = {"dense": q.get("dense_vec"), "colbert": q.get("colbert_vecs")}
            if q.get("lexical_weights") is not None:
                vectors["sparse"] = self.sparse(q["lexical_weights"])
            mode = q["mode"]
            filt = m.Filter.model_validate(q["filter"]) if q.get("filter") else None
            args = dict(
                limit=q["limit"],
                with_payload=True,
                with_vectors=q.get("with_vectors", False),
                query_filter=filt,
            )
            if mode in ("hybrid", "rerank"):
                if mode == "rerank":
                    filt = (
                        m.Filter(must=[m.HasVectorCondition(has_vector="colbert")], should=None)
                        if filt is None
                        else m.Filter(must=[filt, m.HasVectorCondition(has_vector="colbert")])
                    )
                prefetch = [
                    m.Prefetch(query=vectors[v], using=v, filter=filt, limit=q["candidate_limit"])
                    for v in (["dense", "sparse"] if mode == "rerank" else q["modalities"])
                ]
                fusion = m.FusionQuery(fusion=m.Fusion.RRF)
                if mode == "rerank":
                    prefetch = m.Prefetch(
                        prefetch=prefetch, query=fusion, limit=q["candidate_limit"]
                    )
                    args.update(query=vectors["colbert"], using="colbert", prefetch=prefetch)
                else:
                    args.update(query=fusion, prefetch=prefetch)
            else:
                args.update(query=vectors[mode], using=mode)
            points = c.query_points(name, **args).points
        return {"records": [p.model_dump(mode="json", exclude_none=True) for p in points]}

    @staticmethod
    def sparse(weights):
        if any(int(k) > 4294967295 for k in weights):
            raise StorageError(422, "Sparse token ID exceeds uint32")
        pairs = sorted((int(k), v) for k, v in weights.items())
        return m.SparseVector(indices=[k for k, _ in pairs], values=[v for _, v in pairs])
