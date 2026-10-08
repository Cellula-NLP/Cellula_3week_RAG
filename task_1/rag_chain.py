import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()


class RAGPipeline:
    """Builds and executes the RAG chain"""

    def __init__(self, retriever: BaseRetriever, model: str = "nvidia/nemotron-3-super-120b-a12b:free"):
        self.retriever = retriever

        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant acting as a professional knowledge base.
        Answer the user's question based ONLY on the provided context.
        If you do not know the answer based on the context, simply say "I don't have enough information to answer that."

        When answering about projects, experience, or certifications, be SPECIFIC — 
        list the actual project names, companies, or credentials found in the context.

        IMPORTANT: Do NOT include footnote citations like 【1†L1-L3】 in your answer.

        Context: {context}

        Question: {question}
        """)

    def build(self):
        """Constructs the LCEL RAG chain."""

        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown Source")
                formatted.append(f"[Source Document: {source}]\n{doc.page_content}")
            return "\n\n".join(formatted)

        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source