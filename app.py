import streamlit as st
from dotenv import load_dotenv
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from model_prompt import custom_prompt

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def get_models():
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_key, 
        model="text-embedding-3-large"
    )
    vectordb = Chroma(
        persist_directory="Chroma_Indexes", 
        embedding_function=embeddings
    )
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_key,
        max_tokens=512
    )
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return embeddings, vectordb, qa_chain

def copy_to_static_folder(source_path):
    filename = os.path.basename(source_path)
    static_folder_path = os.path.join("static", filename)
    if not os.path.exists(static_folder_path):
        shutil.copy(source_path, static_folder_path)
    return static_folder_path

def main():
    st.set_page_config(page_title="ConsultIQ", layout="wide")
    st.title("ConsultIQ – Skip the Docs. Get the Answers.")

    embeddings, vectordb, qa_chain = get_models()

    query = st.text_input("Ask question about the documents: ", placeholder="Type a question… e.g., “How does PwC approach the automotive sector?")

    if "answer" not in st.session_state:
        st.session_state["answer"] = None
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None
    if "has_queried" not in st.session_state:
        st.session_state["has_queried"] = False

    if st.button("🧠 Ask ConsultIQ") and query:
        st.session_state["has_queried"] = True
        with st.spinner("🤔 Thinking..."):
            result = qa_chain(query)
            st.session_state["answer"] = result

    if st.button("📚 View What ConsultIQ Found") and query:
        st.session_state["has_queried"] = True
        with st.spinner("🔍 Searching relevant content..."):
            results = vectordb.similarity_search_with_score(query)
            st.session_state["search_results"] = results

    if st.session_state["answer"]:
        st.subheader("✨ Generated Result:")
        st.markdown(st.session_state["answer"]['answer'])

        st.subheader("📚 Source Documents:")
        source_documents = st.session_state["answer"].get('source_documents', '[]')

        if source_documents:
            for doc in source_documents:
                doc_dict = doc.dict()
                metadata = doc.metadata
                file_path = metadata.file_path if hasattr(metadata, 'file_path') else "Unknown file"
                page = metadata.page if hasattr(metadata, 'page') else "Unknown page"
                st.markdown(f"- **File:** `{file_path}` | **Page: ** {page}")
        else:
            st.write("No source documents found.")

    if st.session_state["search_results"]:
        st.subheader("🗂️ Top matching relevant documents:")
        for i, (doc, score) in enumerate(st.session_state["search_results"]):
            with st.expander(f"Result {i+1}: (Relevance Score: {score:.2f})", expanded=False):
                text = doc.page_content
                
                if len(text) > 500:
                    text = text[:500] + "..."
                st.markdown(text)
                    
                if 'source' in doc.metadata:
                    file_path = copy_to_static_folder(doc.metadata['source'])
                    file_name = file_path.split("/")[-1]
                st.markdown(f"[{file_name}]({file_path})", unsafe_allow_html=True)
                st.markdown("---")
    
    if st.session_state["has_queried"] and st.session_state["search_results"] is None and st.session_state["answer"] is None:
        st.info("No relevant documents found. ConsultIQ Model won't be able to generate results for you.")

if __name__ == "__main__":
    main()