from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
        You are a helpful AI assistant with expertise in consulting.

        Use the following summaries to answer the question as accurately and concisely as possible.

        Please analyze the summaries and questions in detail. Provide a comprehensive answer of at least 2 paragraphs with 20+ lines of reasoning, using insights from internal documents if needed.
        
        Summaries:
        {summaries}

        Question:
        {question}

        Please provide your answer with references to the source documents if possible.
        End your answer clearly without leaving incomplete sentences or parentheses.
    """
)