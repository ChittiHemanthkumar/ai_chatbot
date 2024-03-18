import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from fpdf import FPDF
import requests 
from fpdf import FPDF
def fetch_url_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to fetch URL: {url}. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching URL: {url}. {e}")

def convert_to_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content)
    return pdf

 

 
load_dotenv()
 
def main():
   
    bg_image = "background_image.jpeg"
    st.image(bg_image, use_column_width=True)


    
    st.header("LLM-powered PDF and URL Chatbot 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
   
    #
    #
    #
    
    st.subheader("Enter the URLs:")

    urls = st.text_area("Enter URLs (one URL per line)")

    if st.button("Convert to PDF"):
        if urls.strip() == "":
            st.warning("Please enter at least one URL.")
        else:
            urls_list = urls.split("\n")
            for index, url in enumerate(urls_list):
                content = fetch_url_content(url)
                if content:
                    pdf = convert_to_pdf(content.decode('utf-8'))
                    st.write(f"PDF generated for URL {index + 1}")
                    st.write(pdf.output(dest="S").encode("utf-8"), format="pdf", width=700, height=500, key=f"pdf_{index}")


     # 
     # 
     #
    # st.write(pdf)
    st.text("Made by CHITTI HEMANTH KUAMR")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings(openai_api_key="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")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()
    
