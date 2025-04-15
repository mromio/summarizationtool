openai_api_key = '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'

from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA


def load_documents(file_path):
    if file_path.endswith('.pdf'): # Check and load the file if in pdf formate
        loader = PyPDFLoader(file_path)
    else: # Check and load the file if in other formats like docx or txt
        loader = UnstructuredFileLoader(file_path)
    return loader.load()

# Split the documents into smaller chunks for analysis
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


# Use OpenAI API key to summarize the documents
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", \
                 openai_api_key = openai_api_key)

def get_summary_chain(strategy="map_reduce"):
    return load_summarize_chain(llm, chain_type=strategy)

def summarize_docs(splits):
    chain = get_summary_chain("map_reduce")
    return chain.run(splits)


# Transform text documents into numerical vectors that capture their semantic meaning
# Enable semantic search to find documents based on meaning instead of keyword matching
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# Create a chain that can answer questions based on the vectorstore
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Load and summarizes the document
docs = load_documents("ParentData_CoverLetter.docx")
splits = split_docs(docs)
summary = summarize_docs(splits)
print("Summary of the document:")
print(summary)

# Highlight the risks associated with the project if applicable to the document
qa = create_qa_chain(create_vectorstore(splits))
answer = qa.run("What are the key risks mentioned in this document?")
print("Answer about risks:")
print(answer)

# Summarize the goal of the project if applicable to the document
qa.run("What are the main goal of the project?")
print("Goals of the project:")
print(answer)
