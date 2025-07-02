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
from langchain.agents import Tool, initialize_agent, AgentType
from agent_tools import search_docs, summarize_text, format_strategy

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

def get_agent():
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_key,
        max_tokens=800
    )

    tools=[
        Tool(
            name="SearchDocs", 
            func=search_docs, 
            description="Useful for retrieving relevant text from internal documents. Use this as the FIRST step for any query."
        ),
        Tool(
            name="Summarize", 
            func=summarize_text, 
            description="Takes raw text from documents (SearchDocs) and summarizes it into bullet points for executives."
        ),
        Tool(
            name="FormatStrategy", 
            func=format_strategy, 
            description="Takes bullet points from Summarize and turns them into a formal multi-paragraph strategic answer."
        )
    ]
    
    agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    return agent

def main():
    st.set_page_config(page_title="ConsultIQ", layout="wide")
    st.title("ConsultIQ – Skip the Docs. Get the Answers.")
    query = st.text_input("Ask question about the documents: ", placeholder="Type a question… e.g., “How does PwC approach the automotive sector?")

    st.subheader("Choose a model:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔧 Simple Model"):
            st.session_state["selected_model"] = "simple"
    with col2:
        if st.button("🔬 Multi-Agent AI Model"):
            st.session_state["selected_model"] = "multi-agent"

    selected_model = st.session_state.get("selected_model", None)

    if selected_model == "simple":
        st.markdown("### 🔧 Simple Model Mode")

        embeddings, vectordb, qa_chain = get_models()

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
                    metadata = doc.metadata
                    file_path = metadata.get('file_path', "Unknown file")
                    page = metadata.get('page', "Unknown page")
                    if file_path != "Unknown file" and os.path.exists(file_path):
                        static_file_path = copy_to_static_folder(file_path)
                        file_name = os.path.basename(static_file_path)
                        url_path = f"/{static_file_path}"
                        st.markdown(f"📄 [{file_name}]({url_path}) Page: {page}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"📄 {file_path} (file not found)")
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
                    st.markdown(f"📄 [{file_name}]({file_path})", unsafe_allow_html=True)
                    st.markdown("---")
        
        if st.session_state["has_queried"] and st.session_state["search_results"] is None and st.session_state["answer"] is None:
            st.info("No relevant documents found. ConsultIQ Model won't be able to generate results for you.")
    
    elif selected_model == "multi-agent":
        st.markdown("### 🔬 Multi-Agent AI Model")

        with st.spinner("🧠 Multi-agent agents collaborating..."):
            agent = get_agent()
            result = agent.run(f"Create a strategic analysis based on internal documents for: '{query}'")
            st.markdown(result)

    else:
        st.markdown("### Please select a model above to get started.")

if __name__ == "__main__":
    main()