import numpy as np
import pytest

from berlinduck.similarity import l2_normalize
from berlinduck.vectorstore import Document, FaissStore, NumpyStore

BACKENDS = [NumpyStore, FaissStore]


def _corpus():
    embeddings = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    documents = [
        Document(id="a", text="near the tower", metadata={"hotel": "A"}),
        Document(id="b", text="tower view room", metadata={"hotel": "B"}),
        Document(id="c", text="airport shuttle", metadata={"hotel": "C"}),
    ]
    return embeddings, documents


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_ranks_by_cosine(backend):
    embeddings, documents = _corpus()
    store = backend(dimension=3)
    store.add(embeddings, documents)

    query = l2_normalize(np.array([[1.0, 0.05, 0.0]]))[0]
    hits = store.search(query, k=3)

    assert [h.document.id for h in hits] == ["a", "b", "c"]
    assert hits[0].score > 0.99  # query is almost collinear with doc "a"
    assert hits[0].score >= hits[1].score >= hits[2].score


@pytest.mark.parametrize("backend", BACKENDS)
def test_k_larger_than_corpus_is_clamped(backend):
    embeddings, documents = _corpus()
    store = backend(dimension=3)
    store.add(embeddings, documents)
    assert len(store.search(embeddings[0], k=99)) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_dimension_mismatch_raises(backend):
    store = backend(dimension=3)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 4), dtype=np.float32), [Document("x", "x"), Document("y", "y")])


@pytest.mark.parametrize("backend", BACKENDS)
def test_doc_count_mismatch_raises(backend):
    store = backend(dimension=3)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 3), dtype=np.float32), [Document("x", "x")])


@pytest.mark.parametrize("backend", BACKENDS)
def test_persist_and_load_round_trip(backend, tmp_path):
    embeddings, documents = _corpus()
    store = backend(dimension=3)
    store.add(embeddings, documents)
    store.persist(tmp_path)

    reloaded = backend.load(tmp_path)
    assert len(reloaded) == 3

    query = l2_normalize(np.array([[0.0, 0.0, 1.0]]))[0]
    before = store.search(query, k=2)
    after = reloaded.search(query, k=2)
    assert [h.document.id for h in before] == [h.document.id for h in after]
    assert after[0].document.metadata == {"hotel": "C"}
