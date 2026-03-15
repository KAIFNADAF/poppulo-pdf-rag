import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Making sure the project root is importable when running: streamlit run app/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline import RAGPipeline  # noqa: E402


st.set_page_config(
    page_title="Poppulo PDF RAG Demo",
    page_icon="📄",
    layout="wide",
)


MAX_DOCUMENTS = 3


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Centering the title and subtitle block */
        .hero-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .hero-header h1 {
            margin-bottom: 0.35rem;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .hero-header p {
            margin: 0 auto;
            max-width: 100%;
            font-size: 0.98rem;
            color: #c9d1d9;
            line-height: 1.3;
            white-space: nowrap;
        }

        /* Keeping the footer fixed and centered */
        .custom-footer {
            position: fixed;
            left: 50%;
            bottom: 10px;
            transform: translateX(-50%);
            z-index: 9999;
            background: rgba(12, 17, 28, 0.92);
            padding: 8px 14px;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            font-size: 0.85rem;
            color: #c9d1d9;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }

        /* Leaving space so the footer does not cover the last section */
        .main .block-container {
            padding-bottom: 5rem;
        }

        /* Highlighting the source badge in the evidence cards */
        .source-badge {
            display: inline-block;
            background-color: rgba(34, 197, 94, 0.16);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.45);
            border-radius: 8px;
            padding: 3px 8px;
            font-size: 0.86rem;
            font-weight: 600;
            margin-top: 0.15rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()

    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False

    if "index_stats" not in st.session_state:
        st.session_state.index_stats = None

    if "indexed_doc_name" not in st.session_state:
        st.session_state.indexed_doc_name = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    if "selected_doc_name" not in st.session_state:
        st.session_state.selected_doc_name = None


def reset_index_state() -> None:
    st.session_state.is_indexed = False
    st.session_state.index_stats = None
    st.session_state.indexed_doc_name = None
    st.session_state.last_result = None


def reset_active_document_state() -> None:
    """
    Resetting both the UI state and the in-memory pipeline state
    when switching the active document.
    """
    reset_index_state()
    st.session_state.pipeline.reset_active_document()


def save_uploaded_pdf_bytes(file_bytes: bytes, file_name: str) -> Path:
    suffix = Path(file_name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        return Path(tmp_file.name)


def get_selected_doc_record():
    selected_name = st.session_state.get("selected_doc_name")
    for doc in st.session_state.uploaded_docs:
        if doc["name"] == selected_name:
            return doc
    return None


def add_uploaded_documents(uploaded_files) -> None:
    """
    Adding newly uploaded files into session state while keeping the
    document list capped at MAX_DOCUMENTS and skipping duplicates by name.
    """
    existing_names = {doc["name"] for doc in st.session_state.uploaded_docs}

    for uploaded_file in uploaded_files:
        if uploaded_file.name in existing_names:
            continue

        if len(st.session_state.uploaded_docs) >= MAX_DOCUMENTS:
            break

        st.session_state.uploaded_docs.append(
            {
                "name": uploaded_file.name,
                "bytes": uploaded_file.getvalue(),
            }
        )
        existing_names.add(uploaded_file.name)

    if st.session_state.uploaded_docs and not st.session_state.selected_doc_name:
        st.session_state.selected_doc_name = st.session_state.uploaded_docs[0]["name"]


def remove_document(doc_name: str) -> None:
    was_selected = st.session_state.selected_doc_name == doc_name
    was_indexed = st.session_state.indexed_doc_name == doc_name

    st.session_state.uploaded_docs = [
        doc for doc in st.session_state.uploaded_docs if doc["name"] != doc_name
    ]

    if was_selected:
        st.session_state.selected_doc_name = (
            st.session_state.uploaded_docs[0]["name"]
            if st.session_state.uploaded_docs
            else None
        )

    if was_indexed:
        reset_active_document_state()

    if not st.session_state.uploaded_docs:
        st.session_state.selected_doc_name = None
        reset_active_document_state()


def get_display_document_name(raw_name: str) -> str:
    return raw_name


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-header">
            <h1>📄 Poppulo PDF RAG Demo</h1>
            <p>Upload a PDF, index it, and ask questions about the content. Answers are generated from the document with supporting sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status() -> None:
    st.subheader("System Status")

    pipeline = st.session_state.pipeline

    try:
        provider_ok = pipeline.healthcheck()
        if provider_ok:
            st.success("🟢 LLM provider is configured and ready.")
        else:
            st.warning("🟠 LLM provider healthcheck did not pass.")
    except Exception as exc:
        st.error(f"🔴 Provider check failed: {exc}")

    if st.session_state.selected_doc_name:
        st.info(f"📄 Selected document: **{st.session_state.selected_doc_name}**")
    else:
        st.info("📄 No document selected yet.")

    if st.session_state.is_indexed and st.session_state.indexed_doc_name:
        display_name = get_display_document_name(st.session_state.indexed_doc_name)
        st.success(f"✅ Indexed document: **{display_name}**")
    else:
        st.info("📦 No document indexed yet.")


def render_sidebar() -> None:
    st.sidebar.header("Document Setup")

    uploaded_files = st.sidebar.file_uploader(
        "Upload up to 3 PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload up to 3 PDFs. You can then choose one active document to index and query.",
    )

    if uploaded_files:
        add_uploaded_documents(uploaded_files)

    doc_count = len(st.session_state.uploaded_docs)

    st.sidebar.caption(f"Stored documents: {doc_count}/{MAX_DOCUMENTS}")

    if doc_count == 0:
        st.sidebar.write("Upload up to 3 PDFs to begin.")
        return

    if doc_count >= MAX_DOCUMENTS:
        st.sidebar.info("Maximum of 3 documents reached. Remove one to add another.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Library")

    doc_names = [doc["name"] for doc in st.session_state.uploaded_docs]

    previous_selection = st.session_state.selected_doc_name
    selected_doc_name = st.sidebar.radio(
        "Choose active document",
        options=doc_names,
        index=doc_names.index(previous_selection) if previous_selection in doc_names else 0,
    )

    # Resetting the active pipeline state when switching the selected document
    if selected_doc_name != st.session_state.selected_doc_name:
        st.session_state.selected_doc_name = selected_doc_name
        reset_active_document_state()

    st.sidebar.write(f"Selected file: **{st.session_state.selected_doc_name}**")

    with st.sidebar.expander("Manage uploaded documents", expanded=False):
        for doc_name in doc_names:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(doc_name)
            with col2:
                if st.button("✕", key=f"remove_{doc_name}"):
                    remove_document(doc_name)
                    st.rerun()

    if st.sidebar.button("Index Selected Document", use_container_width=True):
        selected_doc = get_selected_doc_record()

        if selected_doc is None:
            reset_active_document_state()
            st.sidebar.error("Please select a valid document before indexing.")
            return

        temp_pdf_path = None

        with st.spinner("Parsing, chunking, embedding, and indexing the PDF..."):
            try:
                # Saving the selected PDF to a temporary path before indexing
                temp_pdf_path = save_uploaded_pdf_bytes(
                    selected_doc["bytes"],
                    selected_doc["name"],
                )

                stats = st.session_state.pipeline.index_pdf(
                    temp_pdf_path,
                    document_name=selected_doc["name"],
                )

                stats["document_name"] = selected_doc["name"]

                st.session_state.is_indexed = True
                st.session_state.index_stats = stats
                st.session_state.indexed_doc_name = selected_doc["name"]
                st.session_state.last_result = None

                st.sidebar.success("Document indexed successfully.")
                st.rerun()

            except ValueError as exc:
                reset_active_document_state()
                st.sidebar.error(str(exc))

            except RuntimeError as exc:
                reset_active_document_state()
                st.sidebar.error(f"Indexing failed: {exc}")

            except Exception as exc:
                reset_active_document_state()
                st.sidebar.error(f"Unexpected indexing error: {exc}")

            finally:
                # Cleaning up the temporary file after indexing finishes
                if temp_pdf_path and temp_pdf_path.exists():
                    try:
                        os.remove(temp_pdf_path)
                    except OSError:
                        pass

    if st.session_state.index_stats:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Index Summary")
        stats = st.session_state.index_stats

        st.sidebar.metric("Raw Blocks", stats.get("raw_blocks", 0))
        st.sidebar.metric("Cleaned Blocks", stats.get("cleaned_blocks", 0))
        st.sidebar.metric("Citation Units", stats.get("citation_units", 0))
        st.sidebar.metric("Retrieval Chunks", stats.get("retrieval_chunks", 0))
        st.sidebar.metric("FAISS Vectors", stats.get("faiss_vectors", 0))


def render_question_section() -> None:
    st.subheader("Ask Questions About the Document")

    query = st.text_input(
        "Enter your question",
        placeholder="What are the main findings of this document?",
        disabled=not st.session_state.is_indexed,
    )

    ask_clicked = st.button(
        "Ask",
        type="primary",
        disabled=not st.session_state.is_indexed,
    )

    if ask_clicked:
        clean_query = query.strip()

        if not clean_query:
            st.warning("Please enter a question before asking.")
            return

        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                result = st.session_state.pipeline.answer_query(clean_query)
                st.session_state.last_result = result

            except ValueError as exc:
                st.session_state.last_result = None
                st.error(str(exc))

            except RuntimeError as exc:
                st.session_state.last_result = None
                message = str(exc)

                if "empty response" in message.lower():
                    st.error("The language model returned an empty answer. Please try again.")
                elif "groq generation failed" in message.lower():
                    st.error("The Groq provider failed while generating a response. Please retry.")
                elif "ollama generation failed" in message.lower():
                    st.error(
                        "The Ollama provider failed while generating a response. "
                        "Please make sure Ollama is running and reachable."
                    )
                elif "timed out" in message.lower():
                    st.error("The model request timed out. Please try again.")
                else:
                    st.error(f"Query failed: {message}")

            except Exception as exc:
                st.session_state.last_result = None
                st.error(f"Unexpected query error: {exc}")


def render_answer_section() -> None:
    result = st.session_state.last_result
    if not result:
        return

    st.markdown("---")
    st.subheader("Answer")

    with st.container(border=True):
        st.write(result.get("answer", "No answer returned."))

    supporting_citations = result.get("supporting_citations", [])
    retrieved_results = result.get("retrieved_results", [])
    prompt = result.get("prompt", "")

    st.markdown("---")
    st.subheader("Supporting Evidence")
    st.caption(
        "Top source passages used to support the answer, with document name and page number."
    )

    if supporting_citations:
        top_evidence = supporting_citations[:3]
        extra_evidence = supporting_citations[3:]

        for idx, citation in enumerate(top_evidence, start=1):
            with st.container(border=True):
                display_name = get_display_document_name(citation.document_name)
                st.markdown(f"**Evidence {idx}**")
                st.markdown(
                    f'<div class="source-badge">📄 {display_name} • Page {citation.page_number}</div>',
                    unsafe_allow_html=True,
                )

                snippet = citation.text.strip()
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                st.write(snippet)

        if extra_evidence:
            with st.expander("View more evidence"):
                for idx, citation in enumerate(extra_evidence, start=4):
                    with st.container(border=True):
                        display_name = get_display_document_name(citation.document_name)
                        st.markdown(f"**Evidence {idx}**")
                        st.markdown(
                            f'<div class="source-badge">📄 {display_name} • Page {citation.page_number}</div>',
                            unsafe_allow_html=True,
                        )

                        snippet = citation.text.strip()
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."
                        st.write(snippet)
    else:
        st.info("No strong supporting evidence was available for display.")

    with st.expander("Retrieved Chunks (Debug View)"):
        if retrieved_results:
            for idx, item in enumerate(retrieved_results, start=1):
                display_name = get_display_document_name(item.document_name)

                if item.score is not None:
                    st.markdown(
                        f"**Chunk {idx}** — `{display_name}` | "
                        f"Pages: {', '.join(map(str, item.page_numbers))} | "
                        f"Score: {item.score:.4f}"
                    )
                else:
                    st.markdown(
                        f"**Chunk {idx}** — `{display_name}` | "
                        f"Pages: {', '.join(map(str, item.page_numbers))}"
                    )

                st.write(item.text)

                if item.citation_ids:
                    st.caption(f"Citation IDs: {', '.join(item.citation_ids)}")

                st.markdown("---")
        else:
            st.write("No retrieved chunks available.")

    with st.expander("Prompt (Debug)"):
        st.code(prompt if prompt else "No prompt available.", language="text")


def render_footer() -> None:
    st.markdown(
        '<div class="custom-footer">PDF RAG Demo • Built for grounded document question answering</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    inject_custom_css()
    render_header()
    render_status()
    render_sidebar()
    render_question_section()
    render_answer_section()
    render_footer()


if __name__ == "__main__":
    main()