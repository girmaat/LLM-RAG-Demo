from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap

def build_lcel_chain(llm, retriever):
    # Prompt uses "query" instead of "question"
    prompt_template = """
    You are a helpful assistant that answers user input based on the provided context.

    Use ONLY the information from the context to answer. 
    If you are unsure, say "I don't know".

    Context:
    {context}

    Query:
    {query}

    Helpful Answer
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Extract query from input
    get_query = lambda x: x["query"]
    retriever_step = RunnableLambda(get_query) | retriever

    # Prepare inputs for prompt
    format_input = RunnableMap({
        "context": retriever_step,
        "query": get_query
    })

    # Format into flat string for prompt
    def merge_docs(inputs):
        unique_pages = sorted({doc.metadata['page_number'] for doc in inputs["context"]})
        return {
            "context": "\n\n".join(doc.page_content for doc in inputs["context"]),
            "pages": f"[Pages {', '.join(map(str, unique_pages))}]",  # e.g. [Pages 1, 3, 5]
            "query": inputs["query"]
        }

    # Wrap parsed result + sources
    def wrap_output(inputs):
        return {
            "result": inputs["answer"],
            "source_documents": inputs["documents"],
            "page_numbers": inputs.get("pages", "")
        }

    # Final LCEL RAG Chain
    rag_chain = (
        format_input
        | RunnableLambda(merge_docs)
        | RunnableMap({
            "answer": prompt | llm | StrOutputParser(),
            "documents": lambda x: x["documents"]
        })
        | RunnableLambda(wrap_output)
    )

    return rag_chain
