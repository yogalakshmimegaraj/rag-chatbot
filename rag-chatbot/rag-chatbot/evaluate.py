"""Measure your chatbot: python evaluate.py
Edit tests.json with 15-20 questions about YOUR documents. A test passes if the answer
contains ALL the expected keywords. For 'unanswerable' questions, expect "don't know".
Use the printed accuracy on your resume (e.g. "answered 17/20 test questions correctly")."""
import json
from rag import RagStore, answer

store = RagStore()
tests = json.load(open("tests.json"))
passed = 0
for t in tests:
    reply, _ = answer(store, t["question"])
    ok = all(k.lower() in reply.lower() for k in t["keywords"])
    passed += ok
    print(("PASS" if ok else "FAIL"), "|", t["question"], "->", reply[:100].replace("\n", " "))
print(f"\nAccuracy: {passed}/{len(tests)} = {passed / len(tests):.0%}")
