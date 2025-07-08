from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap

def build_lcel_chain(llm, retriever):
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.

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

    parser = StrOutputParser()

    # Step 1: Extract question from input dict
    def get_question(x):
        return x["question"]

    # Step 2: Pass the question string to retriever
    retriever_step = RunnableLambda(get_question) | retriever

    # Step 3: Combine retriever and question, and build prompt
    def build_output(inputs):
        # 🧪 Debug prints to confirm runtime data types
        print("🧪 QUERY TYPE:", type(inputs["question"]))         # should be str
        print("🧪 CONTEXT TYPE:", type(inputs["context"]))       # should be list
        if inputs["context"]:
            print("🧪 FIRST DOC TYPE:", type(inputs["context"][0]))  # should be Document

        context_str = "\n\n".join([doc.page_content for doc in inputs["context"]])
        formatted_prompt = prompt.format(
            context=context_str,
            question=inputs["question"]
        )

        raw_response = llm.invoke(formatted_prompt)
        return {
            "result": parser.invoke(raw_response),
            "source_documents": inputs["context"]
        }

    rag_chain = (
        RunnableMap({
            "context": retriever_step,
            "question": get_question
        }) | RunnableLambda(build_output)
    )

    return rag_chain
