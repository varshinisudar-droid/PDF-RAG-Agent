from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 2. Connect to the existing ChromaDB
# --------------------------------------------------

print("\nConnecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="pdf_documents"
)

print(f"Documents in database: {collection.count()}")


# --------------------------------------------------
# 3. Ask a question
# --------------------------------------------------

question = input("\nEnter your question: ")


# --------------------------------------------------
# 4. Convert the question into an embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    [question]
).tolist()


# --------------------------------------------------
# 5. Search for the most relevant chunks
# --------------------------------------------------

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)


# --------------------------------------------------
# 6. Display retrieved chunks
# --------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENT CHUNKS")
print("=" * 60)

for index, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][index]

    print(f"\n--- Result {index + 1} ---")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print("\nText:")
    print(document)