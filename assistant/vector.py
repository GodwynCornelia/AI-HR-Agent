import os
from qdrant_client import QdrantClient, models


class VectorIndex:
    def __init__(self,timeout=60):
        self.client = QdrantClient(url=os.getenv('QDRANT_URL', 'http://127.0.0.1:6333'), timeout=timeout)
        self.collection = os.getenv('QDRANT_COLLECTION', 'hr_dense_v1')

    def ensure(self, dimension):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE))

    def upsert(self, chunks, vectors):
        self.ensure(len(vectors[0]))
        self.client.upsert(self.collection, points=[models.PointStruct(
            id=str(chunk.id), vector=vector, payload={'document_id': str(chunk.document_id)})
            for chunk, vector in zip(chunks, vectors)], wait=True)

    def delete(self, document_id):
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key='document_id', match=models.MatchValue(value=str(document_id)))])), wait=True)

    def search(self, vector, limit):
        if not self.client.collection_exists(self.collection):
            return []
        return self.client.query_points(self.collection, query=vector, limit=limit).points
