from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator
from qdrant_client import models

Id = Annotated[StrictInt, Field(ge=0, le=18446744073709551615)] | UUID
Vector = Annotated[list[FiniteFloat], Field(min_length=1024, max_length=1024)]
Sparse = dict[
    Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)$")], Annotated[FiniteFloat, Field(ge=0)]
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Id
    dense_vec: Vector
    lexical_weights: Sparse | None = None
    colbert_vecs: Annotated[list[Vector], Field(min_length=1)] | None = None
    document: str | None = None
    metadata: dict | None = None


class Upsert(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique(self):
        if len({record.id for record in self.records}) != len(self.records):
            raise ValueError("Duplicate record IDs")
        return self


class Ids(BaseModel):
    ids: list[Id] = Field(min_length=1, max_length=100)
    with_vectors: bool = False


class Query(BaseModel):
    mode: Literal["dense", "sparse", "colbert", "hybrid", "rerank"] = "dense"
    dense_vec: Vector | None = None
    lexical_weights: Sparse | None = None
    colbert_vecs: Annotated[list[Vector], Field(min_length=1)] | None = None
    modalities: list[Literal["dense", "sparse", "colbert"]] = Field(
        default_factory=lambda: ["dense", "sparse"]
    )
    limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int = Field(default=100, ge=1, le=1000)
    filter: models.Filter | None = None
    with_vectors: bool = False

    @model_validator(mode="after")
    def required_vectors(self):
        selected = (
            self.modalities
            if self.mode == "hybrid"
            else ["dense", "sparse", "colbert"]
            if self.mode == "rerank"
            else [self.mode]
        )
        if self.mode == "hybrid" and (len(set(selected)) != len(selected) or len(selected) < 2):
            raise ValueError("Hybrid requires two or three unique modalities")
        values = {
            "dense": self.dense_vec,
            "sparse": self.lexical_weights,
            "colbert": self.colbert_vecs,
        }
        if any(not values[name] for name in selected):
            raise ValueError("Missing query embeddings")
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must cover limit")
        return self
