"""Runs without any API key: checks chunking, storage and retrieval with a fake embedder."""
import hashlib, shutil, tempfile
from rag import RagStore, Chunk, chunk_text, answer

def fake_embed(texts, task):          # bag-of-words hash -> 64-dim vector
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v)
    return out

def test_chunking():
    c = chunk_text("word " * 600)
    assert len(c) > 1 and all(len(x) <= 1000 for x in c)

def test_retrieval_and_answer():
    d = tempfile.mkdtemp()
    s = RagStore(path=d, embed_fn=fake_embed, name="t")
    s.add([Chunk("Python is a programming language", "a.pdf", 1),
           Chunk("Chennai is a city in Tamil Nadu", "b.pdf", 2)])
    hits = s.search("Which city is in Tamil Nadu Chennai?", k=1)
    assert hits[0]["source"] == "b.pdf" and hits[0]["page"] == 2
    reply, _ = answer(s, "Which city?", llm=lambda p: "stub:" + ("Chennai" in p and "ok" or "bad"))
    assert reply.endswith("ok")
    shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    test_chunking(); test_retrieval_and_answer(); print("offline tests passed")
