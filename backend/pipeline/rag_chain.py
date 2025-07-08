from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    """
    Constructs a RetrievalQA chain using the given LLM and retriever.
    Returns a LangChain RetrievalQA object.
    """
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.

    Use ONLY the information from the context to answer. 
    If you are unsure, say "I don't know" — do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
