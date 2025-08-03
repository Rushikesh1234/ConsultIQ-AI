from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from model_prompt import custom_prompt

from dotenv import load_dotenv
import os
import shutil

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

def copy_to_static_folder(source_path):
    filename = os.path.basename(source_path)
    static_folder_path = os.path.join("static", filename)
    if not os.path.exists(static_folder_path):
        shutil.copy(source_path, static_folder_path)
    return static_folder_path

def search_docs(query:str) -> str:
    #docs = vectorDB.similarity_search(query=query, k=10)
    #return "\n\n".join([doc.page_content for doc in docs])

    results = vectorDB.similarity_search_with_score(query=query, k=5)

    formatted_docs = []
    for doc, score in results:
        metadata = doc.metadata
        file_path = metadata.get('file_path', "Unknown file")
        page = metadata.get('page', "Unknown page")

        # Handle static file copy if file exists
        if file_path != "Unknown file" and os.path.exists(file_path):
            static_file_path = copy_to_static_folder(file_path)
            file_name = os.path.basename(static_file_path)
            url_path = f"/{static_file_path}"
            source_info = f"[📄 {file_name}]({url_path}) Page: {page}"
        else:
            source_info = f"📄 {file_path} Page: {page} (file not found)"

        snippet = doc.page_content[:400].replace("\n", " ") 
        formatted_docs.append(
            f"{source_info} (Score: {score:.2f})\n{snippet}..."
        )

    return "\n\n".join(formatted_docs)

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

