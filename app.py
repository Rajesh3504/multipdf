import streamlit as st
from PyPDF2 import PdfReader
from google.generativeai import embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
import faiss

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the FAISS index
    faiss.write_index(vector_store.index, "faiss_index.index")

    # Save the metadata
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump((vector_store.index_to_docstore_id, vector_store.docstore), f, protocol=pickle.HIGHEST_PROTOCOL)


# def load_faiss_index():
#     # Load the FAISS index
#     index = faiss.read_index("faiss_index.index")
#
#     # Load the metadata
#     with open("faiss_metadata.pkl", "rb") as f:
#         index_to_docstore_id, docstore = pickle.load(f)
#
#     vector_store = FAISS(index=index, index_to_docstore_id=index_to_docstore_id, docstore=docstore)
#     return vector_store

def load_faiss_index():
    try:
        # Load the FAISS index
        index = faiss.read_index("faiss_index.index")

        # Load the metadata
        with open("faiss_metadata.pkl", "rb") as f:
            index_to_docstore_id, docstore = pickle.load(f)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace with your embeddings instance
        embedding_function = embeddings

        vector_store = FAISS(index=index, index_to_docstore_id=index_to_docstore_id, docstore=docstore, embedding_function=embedding_function)
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None




def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = load_faiss_index()  # Load the FAISS index
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("ASkPDF: Chat with PDF Files")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    st.write("""<div style="position: fixed; bottom: 100px; right: 10px; color: gray;">Created by Rajesh</div>""",
             unsafe_allow_html=True)


if __name__ == "__main__":
    main()
