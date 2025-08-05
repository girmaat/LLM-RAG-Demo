from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.backend.config.config import current_config

def build_qa_chain(llm, retriever, company_name=None):
    """Build QA chain with proper input handling"""
    if company_name is None:
        try:
            module = __import__(
                f"backend.config.profiles.{current_config.domain}.prompts",
                fromlist=["COMPANY_NAME"]
            )
            company_name = getattr(module, "COMPANY_NAME", "the company")
        except ImportError:
            company_name = "the company"

    # Create the prompt template with ALL required variables
    qa_template = """As {company_name}'s HR Assistant, answer based on context:
    
    Context:
    {context}
    
    Conversation History:
    {chat_history}
    
    Question: {question}
    
    Answer:"""
    
    # Create prompt with explicit input variables
    prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question", "chat_history", "company_name"]
    )
    
    # Create a wrapper function to prepare inputs
    def wrapped_chain(inputs):
        # Prepare the inputs with defaults
        prepared_inputs = {
            "question": inputs.get("question", ""),
            "chat_history": inputs.get("chat_history", ""),
            "company_name": inputs.get("company_name", "the company")
        }
        
        # Get documents from retriever
        docs = retriever.get_relevant_documents(prepared_inputs["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Combine all inputs for the LLM
        llm_inputs = {
            "context": context,
            "question": prepared_inputs["question"],
            "chat_history": prepared_inputs["chat_history"],
            "company_name": prepared_inputs["company_name"]
        }
        
        # Create and run the LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        result = llm_chain.run(llm_inputs)
        
        return {
            "answer": result,
            "sources": docs
        }
    
    return wrapped_chain