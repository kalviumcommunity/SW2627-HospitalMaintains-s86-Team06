"""Vector Database Setup and Collection Design Module.

Provides connection, collection design, dimension validation, and readback
functionality using ChromaDB for RAG applications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb

logger = logging.getLogger(__name__)

VECTOR_DIMENSION = 1536
COLLECTION_NAME = "rag_chunks"


@dataclass
class VectorRecord:
    """Schema for a record stored in the vector database."""

    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary format."""
        return {
            "id": self.id,
            "vector": list(self.vector),
            "text": self.text,
            "metadata": dict(self.metadata),
        }


class VectorDBStore:
    """Manager for vector database client connection and collections."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        in_memory: bool = True,
    ) -> None:
        """Initialize VectorDBStore client.

        Args:
            persist_directory: Optional path for persistent storage.
            in_memory: If True, uses an ephemeral in-memory Chroma client.
        """
        self.persist_directory = persist_directory
        self.in_memory = in_memory

        if in_memory or not persist_directory:
            self.client = chromadb.Client()
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)

    def create_collection(
        self,
        name: str = COLLECTION_NAME,
        dimension: int = VECTOR_DIMENSION,
        metric: str = "cosine",
    ) -> VectorCollection:
        """Create or retrieve a collection with specified dimension and metric.

        Args:
            name: Collection name.
            dimension: Required vector dimension size (e.g. 1536).
            metric: Distance metric ("cosine", "l2", "ip").

        Returns:
            VectorCollection wrapper instance.
        """
        hnsw_space = "cosine" if metric in ("cosine", "cosine_similarity") else metric
        chroma_collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": hnsw_space, "dimension": str(dimension)},
        )
        return VectorCollection(
            chroma_collection=chroma_collection,
            name=name,
            dimension=dimension,
            metric=metric,
        )

    def get_collection(
        self,
        name: str = COLLECTION_NAME,
        dimension: int = VECTOR_DIMENSION,
    ) -> VectorCollection:
        """Retrieve an existing collection."""
        chroma_collection = self.client.get_collection(name=name)
        return VectorCollection(
            chroma_collection=chroma_collection,
            name=name,
            dimension=dimension,
        )

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        self.client.delete_collection(name=name)


class VectorCollection:
    """Collection wrapper with dimension validation and CRUD operations."""

    def __init__(
        self,
        chroma_collection: Any,
        name: str,
        dimension: int = VECTOR_DIMENSION,
        metric: str = "cosine",
    ) -> None:
        self.collection = chroma_collection
        self.name = name
        self.dimension = dimension
        self.metric = metric

    def validate_vector(self, vector: Sequence[float]) -> None:
        """Validate vector dimension size early before DB insertion.

        Raises:
            ValueError: If vector dimension does not match collection requirement.
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch for collection '{self.name}': "
                f"expected {self.dimension}, got {len(vector)}"
            )

    def upsert(
        self,
        records: Union[VectorRecord, Dict[str, Any], List[Union[VectorRecord, Dict[str, Any]]]],
    ) -> None:
        """Insert or update records in the vector collection.

        Args:
            records: Single record or list of records (VectorRecord or Dict format).
        """
        if not isinstance(records, list):
            records_list = [records]
        else:
            records_list = records

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in records_list:
            if isinstance(item, VectorRecord):
                rec_id = item.id
                vec = item.vector
                txt = item.text
                meta = item.metadata
            elif isinstance(item, dict):
                rec_id = item["id"]
                vec = item["vector"]
                txt = item["text"]
                meta = item.get("metadata", {})
            else:
                raise TypeError(f"Unsupported record type: {type(item)}")

            self.validate_vector(vec)
            ids.append(rec_id)
            embeddings.append([float(x) for x in vec])
            documents.append(txt)
            metadatas.append(dict(meta))

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read back a single stored record by ID.

        Returns:
            Dictionary with keys: id, vector, text, metadata.
        """
        res = self.collection.get(
            ids=[record_id],
            include=["embeddings", "documents", "metadatas"],
        )

        if not res or not res.get("ids") or len(res["ids"]) == 0:
            return None

        vector = res["embeddings"][0] if res.get("embeddings") is not None and len(res["embeddings"]) > 0 else []
        text = res["documents"][0] if res.get("documents") is not None and len(res["documents"]) > 0 else ""
        metadata = res["metadatas"][0] if res.get("metadatas") is not None and len(res["metadatas"]) > 0 else {}

        return {
            "id": res["ids"][0],
            "vector": list(vector),
            "text": text,
            "metadata": metadata,
        }

    def query(
        self,
        query_vector: Sequence[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query nearest neighbor records by query vector.

        Args:
            query_vector: Query embedding list of length matching VECTOR_DIMENSION.
            n_results: Number of nearest neighbors to return.
            where: Optional metadata metadata filter.

        Returns:
            List of result dictionaries containing id, vector, text, metadata, distance.
        """
        self.validate_vector(query_vector)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [[float(x) for x in query_vector]],
            "n_results": n_results,
            "include": ["embeddings", "documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        res = self.collection.query(**kwargs)
        results: List[Dict[str, Any]] = []

        if res and res.get("ids") and len(res["ids"][0]) > 0:
            ids = res["ids"][0]
            embeddings = res.get("embeddings", [[]])[0]
            documents = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]

            for i in range(len(ids)):
                results.append(
                    {
                        "id": ids[i],
                        "vector": list(embeddings[i]) if i < len(embeddings) else [],
                        "text": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else None,
                    }
                )

        return results

    def count(self) -> int:
        """Return total count of records in the collection."""
        return self.collection.count()
