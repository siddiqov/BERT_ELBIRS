# Main function
import streamlit as st

def main():
    st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")

    st.title("AI Receptionist")
    st.markdown("---")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        ## Sidebar
        Upload your PDF files and ask questions about Location.
    """)
    st.sidebar.markdown("""
        Made with ❤️ by [DanzeeTech](https://www.danzeetech.com/)
    """)
    st.sidebar.markdown("---")

def main_():
    st.set_page_config(page_title="This is the title", page_icon=":robot_face", layout="wide")
    st.title()

#................................................................................................
    language = st.selectbox("Select Language", ["English", "Spanish"])
    model_choice = st.selectbox("Select Model", ["BERT", "RoBERTa", "DistilBERT", "ALBERT"])
    vector_store_type = st.selectbox("Select Vector Store", ["faiss", "elasticsearch"])

    with st.container():
        pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
        if pdf_files:
            with st.spinner('Processing PDFs...'):
                all_chunks, indices, vectors = process_pdfs(pdf_files, language, vector_store_type)

            query = st.text_input(f"How can I help you ({language}):")
            logging.info(f"The user Question: {query}")
            if query:
                results, elapsed_time = measure_performance(all_chunks, indices, vectors, query, vector_store_type)
                st.markdown("---")
                st.subheader("Answer:")
                st.write(results)
                st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()