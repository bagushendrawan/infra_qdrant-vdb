import asyncio
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from src.main import create_app
from src.storage.config import Settings
from src.storage.exceptions import StorageError
from src.storage.schemas import Query, Record, Upsert
from src.storage.worker import Worker


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite(value):
    with pytest.raises(ValidationError):
        Record(id=1, dense_vec=[value] * 1024)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mode": "hybrid", "modalities": ["dense"]},
        {"dense_vec": [1.0]},
        {"mode": "sparse", "lexical_weights": {}},
        {"mode": "rerank", "dense_vec": [1.0] * 1024},
    ],
)
def test_invalid_query(payload):
    with pytest.raises(ValidationError):
        Query(**payload)


def test_duplicate_and_ids():
    record = Record(id=uuid4(), dense_vec=[1.0] * 1024)
    with pytest.raises(ValidationError):
        Upsert(records=[record, record])
    with pytest.raises(ValidationError):
        Record(id="document-1", dense_vec=[1.0] * 1024)


async def test_http(tmp_path):
    app = create_app(Settings(data_path=str(tmp_path)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.get("/health/ready")).status_code == 200
            assert (await c.post("/v1/collections?name=test")).status_code == 201
            assert (await c.post("/v1/collections?name=test")).status_code == 409
            assert (await c.get("/v1/collections/missing")).status_code == 404
            uid = str(uuid4())
            r = await c.post(
                "/v1/collections/test/records/upsert",
                json={"records": [{"id": uid, "dense_vec": [1.0] * 1024}]},
            )
            assert r.status_code == 200
            assert r.headers["X-Request-ID"]
            r = await c.post("/v1/collections/test/query", json={"dense_vec": [1.0] * 1024})
            assert r.json()["records"][0]["id"] == uid
            assert (await c.delete("/v1/collections/test")).status_code == 200


async def test_body_limit():
    app = create_app(Settings(max_bytes=10))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.post("/v1/collections", content=b"x" * 11)).status_code == 413


async def test_timeout_does_not_cancel_accepted_operation():
    worker = Worker(Settings(timeout=0.01))
    worker.ready = True
    completed = asyncio.Event()

    async def slow(*args):
        try:
            await asyncio.sleep(0.05)
            completed.set()
            return {}
        finally:
            worker.busy = False

    worker._run = slow
    with pytest.raises(StorageError) as error:
        await worker.call("list")
    assert error.value.status == 504
    with pytest.raises(StorageError) as error:
        await worker.call("list")
    assert error.value.status == 503
    await asyncio.wait_for(completed.wait(), 1)
    assert not worker.busy


async def test_cancellation_and_shutdown(tmp_path):
    worker = Worker(Settings(data_path=str(tmp_path)))
    await worker.start()
    task = asyncio.create_task(worker.call("create", "survives"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await worker.close()
    restarted = Worker(Settings(data_path=str(tmp_path)))
    await restarted.start()
    try:
        result = await restarted.call("list")
        assert "survives" in result["collections"]
    finally:
        await restarted.close()
