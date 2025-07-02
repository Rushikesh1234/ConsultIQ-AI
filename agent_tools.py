from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from model_prompt import custom_prompt

from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_type=openai_key,
    model="text-embedding-3-large"
)

vectorDB = Chroma(
    persist_directory="Chroma_Indexes",
    embedding_function=embeddings
)

llm = OpenAI(
    temperature=0,
    openai_api_key=openai_key,
    max_tokens=800
)

def search_docs(query:str) -> str:
    docs = vectorDB.similarity_search(query=query, k=10)
    return "\n\n".join([doc.page_content for doc in docs])

def summarize_text(text:str) -> str:
    prompt = custom_prompt.format(
        summaries=text, 
        question="""
        Summarize all key insights from the above text into exhaustive bullet points, suitable for executive analysis.
        Include as many points as necessary to reflect all important aspects from the documents.
        """
    )
    return llm.predict(prompt)

def format_strategy(points:str) -> str:
    prompt = f"""
You are a business strategist. Based on the key points below, write a detailed strategic recommendation.
- Use at least 2 paragraphs (15+ lines total).
- Structure your strategy with a clear beginning, middle, and end.

Key Points:
{points}
"""
    return llm.predict(prompt)

