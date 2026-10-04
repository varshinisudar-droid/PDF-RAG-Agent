import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import search_documents, calculator


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Connect to Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# --------------------------------------------------
# 3. Clean Gemini response
# --------------------------------------------------

def clean_response(response):

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "".join(text_parts).strip()

    return str(content).strip()


# --------------------------------------------------
# 4. Agent
# --------------------------------------------------

def run_agent(user_request):

    # --------------------------------------------------
    # Ask Gemini which tool to use
    # --------------------------------------------------

    decision_prompt = f"""
You are a simple routing agent.

You have exactly TWO tools:

1. DOCUMENT_SEARCH
   Use this for questions related to the
   information contained in the PDF documents.

   The PDFs contain study material about:
   - Cloud Computing
   - Distributed Systems
   - Deep Learning
   - related computer science topics

2. CALCULATOR
   Use this ONLY for mathematical calculations.

Rules:

- If the user asks a mathematical calculation,
  choose CALCULATOR.

- Otherwise, choose DOCUMENT_SEARCH.

- There is NO direct/general-answer option.

Return ONLY one of these exact words:

DOCUMENT_SEARCH
CALCULATOR

User request:
{user_request}
"""

    decision_response = llm.invoke(decision_prompt)

    decision = clean_response(decision_response).upper()


    # --------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------

    if "CALCULATOR" in decision:

        print("\nAgent decision: CALCULATOR")

        calculation_prompt = f"""
Extract ONLY the mathematical expression
from the user's request.

Return ONLY the expression.

User request:
{user_request}
"""

        expression_response = llm.invoke(calculation_prompt)

        expression = clean_response(expression_response)

        result = calculator(expression)

        return f"The result is: {result}"


    # --------------------------------------------------
    # DOCUMENT SEARCH
    # --------------------------------------------------

    else:

        print("\nAgent decision: DOCUMENT SEARCH")

        # Search the PDF database
        retrieved_information = search_documents(user_request)


        # --------------------------------------------------
        # Ask Gemini whether the retrieved PDF
        # information contains the answer
        # --------------------------------------------------

        relevance_prompt = f"""
Determine whether the retrieved PDF content
contains enough information to answer the
user's question.

Retrieved PDF content:

{retrieved_information}

User question:

{user_request}

Return ONLY:

FOUND

or

NOT_FOUND
"""

        relevance_response = llm.invoke(relevance_prompt)

        relevance = clean_response(relevance_response).upper()


        # --------------------------------------------------
        # Answer using PDF
        # --------------------------------------------------

        if "FOUND" in relevance:

            answer_prompt = f"""
Answer the user's question using ONLY the
information provided in the retrieved PDF content.

Do not use outside knowledge.

Retrieved PDF content:

{retrieved_information}

User question:

{user_request}

Give a clear and concise answer.
"""

            answer_response = llm.invoke(answer_prompt)

            return clean_response(answer_response)


        # --------------------------------------------------
        # Answer not found
        # --------------------------------------------------

        else:

            return "The answer was not found in the provided documents."


# --------------------------------------------------
# 5. Run the agent
# --------------------------------------------------

print("=" * 60)
print("TINY AGENT")
print("=" * 60)

user_request = input("\nEnter your request: ")

answer = run_agent(user_request)


# --------------------------------------------------
# 6. Display final answer
# --------------------------------------------------

print("\n" + "=" * 60)
print("AGENT ANSWER")
print("=" * 60)

print(answer)