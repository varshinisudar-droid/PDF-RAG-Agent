from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load the API key from .env
load_dotenv()


# Connect to Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# Send a test question
response = llm.invoke(
    "Explain RAG in one simple sentence."
)


# Display Gemini's response
print("\nGemini response:")
print(response.content)