from pathlib import Path

import fitz
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Find PDF files
# --------------------------------------------------

data_folder = Path("data")
pdf_files = list(data_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")


# --------------------------------------------------
# 2. Extract text from PDFs
# --------------------------------------------------

all_documents = []

for pdf_file in pdf_files:

    print(f"\nReading: {pdf_file.name}")

    document = fitz.open(pdf_file)

    print(f"Pages: {len(document)}")

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            all_documents.append({
                "text": text,
                "source": pdf_file.name,
                "page": page_number + 1
            })

    document.close()


# --------------------------------------------------
# 3. Split text into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for document in all_documents:

    split_texts = text_splitter.split_text(
        document["text"]
    )

    for chunk in split_texts:

        chunks.append({
            "text": chunk,
            "source": document["source"],
            "page": document["page"]
        })


print("\n" + "=" * 50)
print("CHUNKING COMPLETE")
print("=" * 50)

print(f"PDF pages with text : {len(all_documents)}")
print(f"Total chunks        : {len(chunks)}")


# --------------------------------------------------
# 4. Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 5. Create embeddings
# --------------------------------------------------

print("\nCreating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print(f"Created {len(embeddings)} embeddings.")


# --------------------------------------------------
# 6. Create ChromaDB
# --------------------------------------------------

print("\nCreating ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_documents"
)


# --------------------------------------------------
# 7. Store chunks + embeddings
# --------------------------------------------------

ids = [
    f"chunk_{index}"
    for index in range(len(chunks))
]

documents = [
    chunk["text"]
    for chunk in chunks
]

metadatas = [
    {
        "source": chunk["source"],
        "page": chunk["page"]
    }
    for chunk in chunks
]

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n" + "=" * 50)
print("VECTOR DATABASE COMPLETE")
print("=" * 50)

print(f"Documents stored : {collection.count()}")

print("\nExample metadata:")
print(metadatas[0])

print("\nExample stored chunk:")
print(documents[0][:500])

print("\nRAG DATABASE READY!")