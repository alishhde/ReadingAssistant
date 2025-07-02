from smolagents import Tool
from langchain_community.retrievers import BM25Retriever


class PDFRetrieverTool(Tool):
    name = "pdf_retriever"
    description = (
        "Use this tool to answer ANY question about the content of the uploaded PDF/document."
        "If the user asks about information that could be in the uploaded document, always use this tool first."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The question to answer based on the PDF content.",
        }
    }
    output_type = "string"

    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def forward(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found in the document."
        return "\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
