
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



def main():
    load_dotenv()
    st.set_page_config(page_title="PDF gpt", page_icon=":brain:")
    st.header("PDF gpt 🧾")


    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

        chunks = text_splitter.split_text(text)

        # st.write(chunks)

        embeddings = OpenAIEmbeddings()

        similarity_base = FAISS.from_texts(chunks, embeddings)

        question = st.text_input("Ask pdfBOT 🧠")

        if question:
            important_chunk = similarity_base.similarity_search(question)

            # st.write(important_chunk)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                answer = chain.run(input_documents = important_chunk, question=question)
                print(cb)


            st.write(answer)







if __name__ == "__main__":
    main()