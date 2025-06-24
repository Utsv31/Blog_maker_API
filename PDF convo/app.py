import json
import os
import sys
import boto3
import streamlit as st

# 📦 LangChain imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🔐 AWS Bedrock client setup
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock)

# 📂 Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

# 🧠 Vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# 🧠 LLaMA3 model
def get_llama3_llm():
    llm = Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={'max_gen_len': 512}
    )
    return llm

# 📝 Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words with detailed explanations. 
If you don't know the answer, just say that you don't know — don't make one up.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 🔍 Retrieval-based QA
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# 🎨 Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF with Bedrock")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vectors Updated ✅")

    if st.button("LLaMA3 Output"):
        with st.spinner("Processing your question..."):
            try:
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings
                )
                llm = get_llama3_llm()
                answer = get_response_llm(llm, faiss_index, user_question)
                st.write(answer)
                st.success("Done ✅")
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
