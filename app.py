from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load the PDF
pdf_path = "your_pdf.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))


# Split the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# Create the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Extract text from chunks
texts = [
    chunk.page_content
    for chunk in chunks
]


# Convert text into embeddings
vectors = model.encode(
    texts
)

vectors = np.array(
    vectors
).astype("float32")

print("Embeddings created!")

print("Number of vectors:", len(vectors))

print("Vector size:", vectors.shape[1])


# Create FAISS index
index = faiss.IndexFlatL2(
    vectors.shape[1]
)


# Add embeddings to FAISS
index.add(vectors)

print("FAISS index created!")

print(
    "Number of vectors in FAISS:",
    index.ntotal
)


# Ask a question
question = input(
    "\nAsk a question about the PDF: "
)


# Convert question into an embedding
question_vector = model.encode(
    [question]
)

question_vector = np.array(
    question_vector
).astype("float32")


# Search for the 3 most relevant chunks
distances, indices = index.search(
    question_vector,
    k=3
)


# Display retrieved chunks
print("\nRetrieved Chunks:\n")


for i, index_number in enumerate(
    indices[0],
    start=1
):

    chunk = chunks[index_number]

    print(
        f"--- Chunk {i} ---"
    )

    print(
        chunk.page_content
    )

    page = chunk.metadata.get(
        "page"
    )

    if page is not None:

        print(
            f"Page: {page + 1}"
        )

    print()