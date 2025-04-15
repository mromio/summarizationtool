from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA


class DocumentProcessor:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo", temperature=0):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=openai_api_key)

    def load_documents(self, file_path):
        """
        Load documents from a file.

        Args:
            file_path (str): Path to the document file.

        Returns:    
            list: List of loaded documents.

        """
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        return loader.load()

    def split_docs(self, docs, chunk_size=1000, chunk_overlap=100):
        """Split documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)

    def get_summary_chain(self, strategy="map_reduce"):
        """Get a summarization chain."""
        return load_summarize_chain(self.llm, chain_type=strategy)

    def summarize_docs(self, splits, strategy="map_reduce"):
        """Summarize document splits."""
        chain = self.get_summary_chain(strategy)
        return chain.run(splits)

    def create_vectorstore(self, docs):
        """Create a vectorstore for semantic search."""
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    def create_qa_chain(self, vectorstore):
        """Create a QA chain using the vectorstore."""
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
        return qa_chain


def main(file_path, openai_api_key):
    processor = DocumentProcessor(openai_api_key=openai_api_key)

    # Load and process the document
    docs = processor.load_documents(file_path)
    splits = processor.split_docs(docs)

    # Summarize the document
    summary = processor.summarize_docs(splits)
    print("Summary of the document:")
    print(summary)

    # Create a QA chain and answer questions
    vectorstore = processor.create_vectorstore(splits)
    qa = processor.create_qa_chain(vectorstore)

    risks_answer = qa.run("What are the key risks mentioned in this document?")
    print("Answer about risks:")
    print(risks_answer)

    goals_answer = qa.run("What are the main goals of the project?")
    print("Goals of the project:")
    print(goals_answer)


if __name__ == "__main__":
    # Example usage
    file_path = "Project 2 Plan.pdf"  
    openai_api_key = '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'
    main(file_path, openai_api_key)
