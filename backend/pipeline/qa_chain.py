from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    prompt_template = """
    You are a helpful assistant that answers user input based on the provided context.

    Use ONLY the information from the context to answer. 
    If you are unsure, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain
