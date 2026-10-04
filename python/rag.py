import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="pdf_documents"
)

print(f"Documents in database: {collection.count()}")


# --------------------------------------------------
# 4. Connect to Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# --------------------------------------------------
# 5. Ask the user a question
# --------------------------------------------------

question = input("\nEnter your question: ")


# --------------------------------------------------
# 6. Convert question into an embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    [question]
).tolist()


# --------------------------------------------------
# 7. Retrieve relevant chunks
# --------------------------------------------------

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)


# --------------------------------------------------
# 8. Prepare retrieved context
# --------------------------------------------------

retrieved_chunks = results["documents"][0]
retrieved_metadata = results["metadatas"][0]

context_parts = []

for index, chunk in enumerate(retrieved_chunks):

    metadata = retrieved_metadata[index]

    context_parts.append(
        f"""
Source: {metadata['source']}
Page: {metadata['page']}

Content:
{chunk}
"""
    )


context = "\n".join(context_parts)


# --------------------------------------------------
# 9. Create the RAG prompt
# --------------------------------------------------

prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved document context below.

If the answer is not present in the context, clearly say:
"I could not find the answer in the provided documents."

Do not invent information.

Retrieved document context:
{context}

User question:
{question}
"""


# --------------------------------------------------
# 10. Ask Gemini to generate the answer
# --------------------------------------------------

response = llm.invoke(prompt)


# --------------------------------------------------
# 11. Display the final answer
# --------------------------------------------------

print("\n" + "=" * 60)
print("RAG ANSWER")
print("=" * 60)

if isinstance(response.content, list):
    for item in response.content:
        if item.get("type") == "text":
            print(item.get("text"))
else:
    print(response.content)


# --------------------------------------------------
# 12. Display sources
# --------------------------------------------------

print("\n" + "=" * 60)
print("SOURCES USED")
print("=" * 60)

for index, metadata in enumerate(retrieved_metadata):

    print(
        f"{index + 1}. "
        f"{metadata['source']} - "
        f"Page {metadata['page']}"
    )