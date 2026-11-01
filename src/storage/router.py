from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from fastapi import Query as QueryParam

from src.storage.schemas import Ids, Query, Upsert

router = APIRouter(prefix="/v1/collections", tags=["collections"])
Name = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


async def worker(request: Request):
    return request.app.state.worker


DB = Annotated[object, Depends(worker)]


@router.post("", status_code=201, summary="Create a BGE-M3 collection")
async def create(name: Annotated[str, QueryParam(pattern=r"^[A-Za-z0-9_-]{1,128}$")], db: DB):
    return await db.call("create", name)


@router.get("", summary="List collections")
async def listing(db: DB):
    return await db.call("list")


@router.get("/{name}", summary="Inspect collection")
async def info(name: Name, db: DB):
    return await db.call("info", name)


@router.delete("/{name}", summary="Delete collection and its records")
async def drop(name: Name, db: DB):
    return await db.call("drop", name)


@router.post("/{name}/records/upsert", summary="Replace complete records")
async def upsert(name: Name, payload: Upsert, db: DB):
    return await db.call("upsert", name, payload.model_dump(mode="json"))


@router.post("/{name}/records/get", summary="Fetch records by IDs")
async def get(name: Name, payload: Ids, db: DB):
    return await db.call("get", name, payload.model_dump(mode="json"))


@router.post("/{name}/records/delete", summary="Delete explicit record IDs")
async def delete(name: Name, payload: Ids, db: DB):
    return await db.call("delete", name, payload.model_dump(mode="json"))


@router.post("/{name}/query", summary="Search dense, sparse, ColBERT, fusion or reranking")
async def query(name: Name, payload: Query, db: DB):
    return await db.call("query", name, payload.model_dump(mode="json"))
