from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
        You are a helpful AI assistant with expertise in consulting.

        Use the following summaries to answer the question as accurately and concisely as possible.

        Summaries:
        {summaries}

        Question:
        {question}

        Please provide your answer with references to the source documents if possible.
        End your answer clearly without leaving incomplete sentences or parentheses.
    """
)