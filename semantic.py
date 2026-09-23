"""Optional embeddings. Only eligible descriptions are sent; caches stay in session memory."""
import math


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Invalid embedding dimensions")
    if not all(math.isfinite(v) for v in [*left, *right]):
        raise ValueError("Non-finite embedding")
    norm = math.sqrt(sum(v * v for v in left) * sum(v * v for v in right))
    return sum(a * b for a, b in zip(left, right)) / norm if norm else 0.0


class EmbeddingRanker:
    def __init__(self, client, model="text-embedding-3-small", cache=None):
        self.client = client
        self.model = model
        self.cache = cache if cache is not None else {}

    def __call__(self, query, documents):
        texts = [query, *documents]
        missing = list(dict.fromkeys(t for t in texts if (self.model, t) not in self.cache))
        if missing:
            response = self.client.embeddings.create(model=self.model, input=missing, encoding_format="float")
            rows = sorted(response.data, key=lambda item: item.index)
            if [r.index for r in rows] != list(range(len(missing))):
                raise ValueError("Incomplete embeddings response")
            vectors = [r.embedding for r in rows]
            for vector in vectors:
                cosine(vectors[0], vector)
            # Bound session cache; model name and full text invalidate changed descriptions.
            if len(self.cache) + len(missing) > 512:
                keep = {(self.model, text) for text in texts}
                for key in list(self.cache):
                    if key not in keep:
                        del self.cache[key]
            self.cache.update({(self.model, text): vector for text, vector in zip(missing, vectors)})
        vectors = [self.cache[(self.model, text)] for text in texts]
        return [max(0.0, min(1.0, cosine(vectors[0], vector))) for vector in vectors[1:]]
