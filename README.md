# Qdrant VDB

Persistent embedded Qdrant behind FastAPI, accepting precomputed BGE-M3 embeddings.
No GPU, Docker, embedding models or encoder HTTP calls are required.

```bash
uv sync --locked
uv run python main.py
```

Default URL: http://127.0.0.1:9106. Override `--host` and `--port`.
One spawned process owns the database, and one request executes at a time.
Concurrent calls receive 503 with Retry-After. `/health/live` checks HTTP liveness;
`/health/ready` checks database availability. `/docs` is enabled for local/staging.
Requests carry X-Request-ID; timing logs omit embeddings and document bodies.
No authentication is provided; use a trusted network.

Settings use `QDRANT_VDB_` and `.env`; see `.env.example`. Data defaults to
`./data/qdrant` relative to this service. Back up the directory after stopping
this service. Only one service instance may own a directory. Initialization failure
stops startup. Worker failure makes readiness false; restart the service to recover.

## API

Create a collection with `POST /v1/collections?name=documents`. List with
`GET /v1/collections`; inspect or delete with `GET` or `DELETE /v1/collections/documents`.
Names contain 1–128 letters, digits, underscores or hyphens.
Duplicate creation returns 409; missing collections return 404.

All collections have `dense` (1024 dimensions, cosine), `sparse` (dot product,
no IDF adjustment), and `colbert` (1024 dimensions per token, cosine MaxSim).

Save with `POST /v1/collections/documents/records/upsert`:

```python
# encoded is the response from the BGE-M3 /v1/encode API.
payload = {"records": [{
    "id": 1,
    "dense_vec": encoded["dense_vecs"][0],
    "lexical_weights": encoded["lexical_weights"][0],  # optional
    "colbert_vecs": encoded["colbert_vecs"][0],        # optional
    "document": "Hello world",
    "metadata": {"source": "example"},
}]}
```

IDs are unsigned 64-bit integers or UUID strings. Dense vectors are required;
sparse token maps use numeric token ID strings and nonnegative finite weights.
Upserts replace the entire record, including payload and all embeddings. Omitted
optional fields are removed. Maximum batch size is 16; duplicate IDs are rejected.
Upserts and deletions return acknowledged IDs, including already-absent delete IDs.
Batch writes are not promised to be atomic on interruption; retry complete records
with the same IDs if an interrupted batch needs reconciliation.

Fetch using `POST .../records/get` with `{"ids":[1],"with_vectors":true}`.
Missing IDs are omitted. Results are Qdrant records under `records`; vector keys
are `dense`, `sparse` (indices/values), and `colbert`. Delete using
`POST .../records/delete` with `{"ids":[1]}`.

Search using `POST .../query` with a single query's `dense_vec`, `lexical_weights`,
and/or `colbert_vecs` and these controls:

| Field | Default / behavior |
| --- | --- |
| `mode` | `dense`; also `sparse`, `colbert`, `hybrid`, `rerank` |
| `modalities` | `["dense","sparse"]`; hybrid requires 2–3 unique types |
| `limit` | 10; maximum 100 |
| `candidate_limit` | 100; maximum 1000; must cover limit |
| `filter` | Optional Qdrant payload filter; metadata keys use `metadata.source` etc. |
| `with_vectors` | false; payload is always returned |

Supply all embeddings required by the selected mode. Standalone searches evaluate
the filtered collection independently. Hybrid fuses the selected representations'
candidate lists with Qdrant RRF. Rerank retrieves dense+sparse candidates that have
ColBERT vectors, fuses them, then applies ColBERT to that shortlist. Results contain
`records` with `id`, `score`, and `payload`. Higher scores rank first. Scores are
native Qdrant scores, not comparable across modes: ColBERT sums token MaxSim
scores; BGE-M3's helper uses a mean, giving different magnitudes but the same ordering
for a fixed query. Records lacking a searched representation are excluded for it.

Invalid inputs return 422. Request and result limits default to 64 MiB (413).
Operations time out after 120 seconds (504), but accepted operations continue,
as they do after disconnection. Check state before retrying a timed-out write.
The worker remains busy until completion; shutdown drains accepted work and closes
storage. Unexpected worker failure returns 503 and requires service restart.

## Capacity and verification

Embedded mode persists to disk but retains vectors in memory and searches
exhaustively. 10,000 records is a workload target, not a performance guarantee:
512 token vectors × 1024 dimensions × float32 × 10,000 is about 20 GB before
payloads, dense vectors, and other overhead. ColBERT queries can be slow.
Use a Qdrant server for larger workloads or predictable latency.

```bash
uv run pytest -q
uv run ruff check src tests main.py
```

The shared smoke suite tests all five query modes using synthetic embeddings in
an isolated collection and deletes that collection afterward. It does not require
BGE-M3 for its Qdrant check. Production-scale latency and RAM remain deployment checks.

## Original resources

- [Qdrant Python client](https://github.com/qdrant/qdrant-client) — upstream client and embedded local storage implementation used by this service.
- [Qdrant](https://github.com/qdrant/qdrant) — upstream vector search engine and server deployment option.
- [BAAI/bge-m3 model card](https://huggingface.co/BAAI/bge-m3) — embedding model whose dense, sparse, and ColBERT outputs this API accepts.
