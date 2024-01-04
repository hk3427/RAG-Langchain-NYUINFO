import streamlit as st
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embed import EmbedChunks
import requests
import logging

# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class ChatApp:
    def __init__(self, model_name, chromadb_path):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.chromadb_path = chromadb_path

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        # You can configure the logging format and handlers as needed

    def get_retriever(self):
        embed_chunks_instance = EmbedChunks(model_name=self.model_name)
        vectordb = Chroma(persist_directory=self.chromadb_path,collection_name="test",
                          embedding_function=embed_chunks_instance.embedding_model
                          )
        retriever = vectordb.as_retriever()
        return retriever

    def create_chain(self, retriever):
        llm = LlamaCpp(
            model_path="/Users/himanshukumar/Desktop/nyuinfo_rag/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0,
            verbose=False,
            streaming=True,
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=False
        )

        return qa_chain

    def process_chat(self):
        self.setup_logging()

        st.set_page_config(
            page_title="Your own AI-Chat!"
        )

        st.header("Your own AI-Chat!")

        retriever = self.get_retriever()

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "How may I help you today?"}
            ]

        if "current_response" not in st.session_state:
            st.session_state.current_response = ""

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        llm_chain = self.create_chain(retriever)

        user_prompt = st.chat_input("Your message here", key="user_input")
        if user_prompt:
            st.session_state.messages.append(
                {"role": "user", "content": user_prompt}
            )

            with st.chat_message("user"):
                st.markdown(user_prompt)

            response = llm_chain.run(user_prompt)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    chat_app = ChatApp(model_name="sentence-transformers/all-MiniLM-l6-v2", chromadb_path='/Users/himanshukumar/Desktop/nyuinfo_rag/chromdb/')
    chat_app.process_chat()
