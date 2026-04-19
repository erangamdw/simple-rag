from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

def load_pdf_documents(data_dir: Path):
    docs = []
    for pdf_file in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    return docs

def build_vectorstore():
    documents = load_pdf_documents(DATA_DIR)

    if not documents:
        raise ValueError("No PDF files found in ./data")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def ask_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', 'n/a')}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    llm = ChatOpenAI(model="gpt-4.1-mini")

    prompt = f"""
You are answering questions using only the provided PDF context.
If the answer is not in the context, say you could not find it in the PDFs.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, relevant_docs

def main():
    print("Building vector store from PDFs...")
    vectorstore = build_vectorstore()
    print("Ready.")

    while True:
        question = input("\nAsk a question about the PDFs (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        answer, docs = ask_question(vectorstore, question)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        for i, doc in enumerate(docs, start=1):
            print(
                f"{i}. {doc.metadata.get('source', 'unknown')} "
                f"(page {doc.metadata.get('page', 'n/a')})"
            )

if __name__ == "__main__":
    main()