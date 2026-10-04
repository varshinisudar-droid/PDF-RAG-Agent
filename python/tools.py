import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="pdf_documents"
)


# --------------------------------------------------
# Document Search Tool
# --------------------------------------------------

def search_documents(question):
    """
    Search the PDF database for information
    relevant to the user's question.
    """

    # Convert question into an embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    # Prepare results
    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    output = []

    for index, chunk in enumerate(retrieved_chunks):

        metadata = retrieved_metadata[index]

        output.append(
            f"""
Source: {metadata['source']}
Page: {metadata['page']}

Content:
{chunk}
"""
        )

    return "\n".join(output)

# --------------------------------------------------
# Calculator Tool
# --------------------------------------------------

def calculator(expression):
    """
    Perform basic mathematical calculations.
    """

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Unable to calculate the expression."