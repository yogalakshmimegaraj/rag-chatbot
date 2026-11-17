"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st
from rag import RagStore, answer, load_pdf

st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents")


@st.cache_resource
def get_store():
    return RagStore()


store = get_store()

with st.sidebar:
    st.header("1. Upload PDFs")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Build index", disabled=not files):
        with st.spinner("Reading and embedding..."):
            store.reset()
            total = 0
            for f in files:
                chunks = load_pdf(f, f.name)
                store.add(chunks)
                total += len(chunks)
        st.success(f"Indexed {total} chunks from {len(files)} file(s).")
    st.caption(f"Chunks in index: {store.col.count()}")

st.header("2. Ask questions")
if "history" not in st.session_state:
    st.session_state.history = []

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}, page {s['page']}** (distance {s['distance']:.2f})")
                    st.caption(s["text"][:300] + "...")

if q := st.chat_input("Ask something about your documents"):
    st.session_state.history.append(("user", q, None))
    with st.spinner("Thinking..."):
        try:
            reply, sources = answer(store, q)
        except Exception as e:  # e.g. missing key, wrong model name
            reply, sources = f"Error: {e}", None
    st.session_state.history.append(("assistant", reply, sources))
    st.rerun()
