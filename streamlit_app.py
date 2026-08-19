import streamlit as st
import tempfile
import os
import faiss
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


load_dotenv()


st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)


if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "model" not in st.session_state:
    st.session_state.model = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("📄 Chat with PDF")

st.write(
    "Ask questions and get answers directly from your document."
)


with st.sidebar:

    st.header("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.info(
        "Upload a PDF and ask questions about its content."
    )


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    new_file = (
        st.session_state.file_name
        != uploaded_file.name
    )


    st.session_state.file_name = uploaded_file.name


    if new_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name


        try:

            loader = PyPDFLoader(
                pdf_path
            )

            documents = loader.load()

        except Exception:

            st.error(
                "Could not read the PDF."
            )

            st.stop()


        st.write(
            "Number of pages:",
            len(documents)
        )


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )


        chunks = text_splitter.split_documents(
            documents
        )


        st.write(
            "Number of chunks:",
            len(chunks)
        )


        st.session_state.chunks = chunks


        with st.expander(
            "📖 View Extracted Text"
        ):

            for document in documents:

                st.write(
                    document.page_content
                )


        if st.session_state.model is None:

            st.session_state.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )


        model = st.session_state.model


        texts = [
            chunk.page_content
            for chunk in chunks
        ]


        vectors = model.encode(
            texts
        )


        vectors = np.array(
            vectors
        ).astype("float32")


        st.write(
            "Embeddings created!"
        )


        st.write(
            "Number of vectors:",
            len(vectors)
        )


        st.write(
            "Vector size:",
            vectors.shape[1]
        )


        index = faiss.IndexFlatL2(
            vectors.shape[1]
        )


        index.add(
            vectors
        )


        st.session_state.index = index


        st.write(
            "FAISS index created!"
        )


        st.write(
            "Number of vectors in FAISS:",
            index.ntotal
        )


        os.remove(
            pdf_path
        )


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if "source" in message:

                st.write(
                    f"📄 {message['source']}"
                )


    question = st.chat_input(
        "Ask a question about the PDF..."
    )


    if question:

        with st.chat_message("user"):

            st.write(
                question
            )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        model = st.session_state.model

        index = st.session_state.index

        chunks = st.session_state.chunks


        question_vector = model.encode(
            [question]
        )


        question_vector = np.array(
            question_vector
        ).astype("float32")


        distances, indices = index.search(
            question_vector,
            k=3
        )


        retrieved_chunks = []


        for index_number in indices[0]:

            retrieved_chunks.append(
                chunks[index_number]
            )


        with st.expander(
            "🔍 View Sources"
        ):

            for i, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                page = chunk.metadata.get(
                    "page"
                )


                if page is not None:

                    page = page + 1


                st.write(
                    f"Source {i} — Page {page}"
                )


                st.write(
                    chunk.page_content
                )


        context = "\n\n".join(
            chunk.page_content
            for chunk in retrieved_chunks
        )


        pages = []


        for chunk in retrieved_chunks:

            page = chunk.metadata.get(
                "page"
            )


            if page is not None:

                pages.append(
                    page + 1
                )


        pages = sorted(
            set(pages)
        )


        if pages:

            source_text = (
                "Source pages: "
                + ", ".join(
                    str(page)
                    for page in pages
                )
            )

        else:

            source_text = (
                "Source page unavailable"
            )


        api_key = os.getenv(
            "GROQ_API_KEY"
        )


        if not api_key:

            st.error(
                "GROQ_API_KEY was not found. "
                "Please check your .env file."
            )

            st.stop()


        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key
        )


        prompt = f"""
Answer the question using only the following context.

If the answer is not present in the context,
say:

"I don't know based on the provided PDF."

Do not use outside knowledge.

Context:

{context}

Question:

{question}
"""


        try:

            response = llm.invoke(
                prompt
            )

        except Exception as e:

            st.error(
                "Something went wrong while generating the answer."
            )

            st.write(
                str(e)
            )

            st.stop()


        with st.chat_message(
            "assistant"
        ):

            st.write(
                response.content
            )

            st.write(
                f"📄 {source_text}"
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "source": source_text
            }
        )