#This code is working
import os
import streamlit as st
import sqlite3
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI  # Import from langchain_community instead of langchain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback  # Updated import
from langchain.docstore.document import Document
import hashlib

# Load environment variables
load_dotenv()

# Function to create database tables
def create_tables(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vector_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_name TEXT,
                  vector BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS document_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_name TEXT,
                  document_hash TEXT)''')
    conn.commit()

# Function to compute MD5 hash of file content
def compute_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

# Function to check if document exists in database
def document_exists(conn, document_name, document_hash):
    c = conn.cursor()
    c.execute('SELECT id FROM document_metadata WHERE document_name=? AND document_hash=?', (document_name, document_hash))
    return c.fetchone() is not None

# Function to save document metadata to database
def save_document_metadata(conn, document_name, document_hash):
    c = conn.cursor()
    c.execute('INSERT INTO document_metadata (document_name, document_hash) VALUES (?, ?)', (document_name, document_hash))
    conn.commit()

# Function to create vector store in database
def create_vector_store(chunks, store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        c = conn.cursor()

        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents(chunks)
        
        for vec in vectors:
            vec_np = np.array(vec, dtype=np.float32)
            c.execute('INSERT INTO vector_store (store_name, vector) VALUES (?, ?)', (store_name, sqlite3.Binary(vec_np.tobytes())))
        
        conn.commit()
        conn.close()
        
    except sqlite3.Error as e:
        st.error(f"SQLite error while creating vector store: {e}")

# Function to load vectors from database
def load_vector_store(store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        c = conn.cursor()
        
        c.execute('''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='vector_store' ''')
        if c.fetchone()[0] != 1:
            st.error("Table 'vector_store' does not exist.")
            return []
        
        c.execute('SELECT vector FROM vector_store WHERE store_name=?', (store_name,))
        vectors = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
        
        conn.close()
        
        return vectors
    
    except sqlite3.Error as e:
        st.error(f"SQLite error while loading vector store: {e}")
        return []

# Function to summarize text
def summarize_text(text):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.run(f"Please summarize the following text: {text}")
    return response

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")
    
    # Header
    st.title("Chat With Judicial PDF document")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        ## Sidebar
        Upload your PDF files and ask questions about judicial documents.
        
        Made with ❤️ by [DanzeeTech](https://www.danzeetech.com/)
    """)
    st.sidebar.markdown("---")
    
    # Main content area
    with st.container():
        pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
        
        if pdf_files:
            all_chunks = []
            conn = sqlite3.connect('my_database.db')
            create_tables(conn)
            
            for pdf in pdf_files:
                pdf_reader = PdfReader(pdf)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    length_function=len)
                
                chunks = text_splitter.split_text(text=text)
                
                summarized_chunks = [summarize_text(chunk) if len(chunk.split()) > 500 else chunk for chunk in chunks]
                store_name = pdf.name[:-4]
                file_content = pdf.getvalue()
                document_hash = compute_hash(file_content)
                
                if document_exists(conn, pdf.name, document_hash):
                    st.write(f"Embeddings Loaded from Disk for {pdf.name}")
                    chunks = load_vector_store(store_name)
                else:
                    st.write(f"Embeddings Computation Completed for {pdf.name}")
                    create_vector_store(summarized_chunks, store_name)
                    save_document_metadata(conn, pdf.name, document_hash)
                
                all_chunks.extend(summarized_chunks)
            
            conn.close()

            query = st.text_input("Ask question about your Judicial PDF files:")
            if query:
                docs = [Document(page_content=chunk) for chunk in all_chunks]
                
                llm = OpenAI(temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="map_reduce")
                
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    st.markdown("---")
                    st.subheader("Answer:")
                    st.write(response)
                    st.markdown("---")
                    st.write(f"Total Tokens: {cb.total_tokens}")
                    st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.write(f"Completion Tokens: {cb.completion_tokens}")
                    st.write(f"Total Cost (USD): ${cb.total_cost:.5f}")

if __name__ == '__main__':
    main()
