import requests
import streamlit as st

st.set_page_config(page_title="Enterprise Knowledge Copilot", page_icon="📚", layout="wide")
API = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000").rstrip("/")
st.title("📚 AI Enterprise Knowledge Copilot")
st.caption("Upload organizational documents and ask grounded questions with source references.")

with st.sidebar:
    st.subheader("Upload documents")
    uploaded = st.file_uploader("PDF, TXT, or Markdown", type=["pdf", "txt", "md"])
    if st.button("Upload and index", disabled=uploaded is None, use_container_width=True):
        try:
            response = requests.post(f"{API}/documents/upload",
                                     files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                                     timeout=120)
            if response.ok:
                st.success(f"Indexed {response.json()['document']} ({response.json()['chunks']} chunks)")
            else:
                st.error(response.text)
        except requests.RequestException as e:
            st.error(f"Backend connection failed: {e}")
    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json().get("documents", [])
        if docs:
            for doc in docs:
                st.write(f"**{doc['name']}** — {doc['chunks']} chunks")
        else:
            st.info("No documents indexed yet.")
    except requests.RequestException:
        st.warning("Start the FastAPI backend first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {src['document']}**, page {src.get('page') or 'N/A'} · score {src['score']}")
                    st.caption(src["text"])

question = st.chat_input("Ask about your uploaded documents…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = st.session_state.messages[:-1][-8:]
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                res = requests.post(f"{API}/chat", json={"question": question, "history": history}, timeout=90)
                res.raise_for_status()
                data = res.json()
                st.markdown(data["answer"])
                st.caption(f"Answer mode: {data['mode']}")
                with st.expander("Retrieval and answer quality"):
                 st.write("**Search query:**")
                 st.write(data.get("retrieval_query", question))

                 st.write("**Citation check:**")
                 st.json(data.get("citation_check", {}))

                 st.write("**Grounding check:**")
                 st.json(data.get("grounding_check", {}))

                if data["sources"]:
                    with st.expander("Sources"):
                        for i, src in enumerate(data["sources"], 1):
                            st.markdown(f"**[{i}] {src['document']}**, page {src.get('page') or 'N/A'} · score {src['score']}")
                            st.caption(src["text"])
                st.session_state.messages.append({"role": "assistant", "content": data["answer"], "sources": data["sources"]})
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Request failed: {e}"})

st.divider()
st.subheader("Document tools")
try:
    docs = requests.get(f"{API}/documents", timeout=5).json().get("documents", [])
    names = [d["name"] for d in docs]
except Exception:
    names = []
c1, c2 = st.columns(2)
with c1:
    selected = st.selectbox("Summarize a document", names, key="summary_doc") if names else None
    if st.button("Generate summary", disabled=not selected):
        try:
            r = requests.post(f"{API}/documents/{selected}/summary", timeout=120)
            r.raise_for_status()
            st.markdown(r.json()["summary"])
        except Exception as e:
            st.error(str(e))
with c2:
    if len(names) >= 2:
        a = st.selectbox("Document A", names, key="compare_a")
        b = st.selectbox("Document B", names, index=1 if len(names)>1 else 0, key="compare_b")
        if st.button("Compare documents", disabled=a == b):
            try:
                r = requests.post(f"{API}/documents/compare", json={"document_a": a, "document_b": b}, timeout=120)
                r.raise_for_status()
                st.markdown(r.json()["comparison"])
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Index at least two documents to compare them.")
