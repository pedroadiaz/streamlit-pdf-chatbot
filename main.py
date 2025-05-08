import streamlit as st
import os

from click import option
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.messages import AIMessage
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Directory to store PDFs
pdf_dir = "./pdfs/"
os.makedirs(pdf_dir, exist_ok=True)

# template for answering questions
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="ConventionalLoans.pdf" )
model = ChatOpenAI(model="gpt-4o-mini")

# Chat engine initialization
llm_engine = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management for chat
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I Am your Ai. How can I help you ? 💻"}
    ]


# Custom CSS styling
def apply_custom_css():
    st.markdown(
        """
    <style>
        .main {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
        }
        .stTextInput textarea {
            color: #ffffff !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            color: white !important;
            background-color: #3d3d3d !important;
        }
        .stSelectbox svg {
            fill: white !important;
        }
        .stSelectbox option {
            background-color: #2d2d2d !important;
            color: white !important;
        }
        div[role="listbox"] div {
            background-color: #2d2d2d !important;
            color: white !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Sidebar configuration
def configure_sidebar():
    with st.sidebar:
        st.header("⚙️ File Learner Chat Bot ")
        # selected_model = st.selectbox(
        #     "Choose Model", ["deepseek-r1:7b", "deepseek-r1:7b"], index=0
        # )
        st.divider()
        st.markdown("## Model Capabilities")
        st.markdown(
            """

        - 🐞 Debugging Assistant
        - 🐞 Quick Reader
        - 📝 Code Documentation
        - 💡 Solution Design
        """
        )
        st.divider()
        st.markdown(
            "Built with [OpenAI](https://openai.com/) | [LangChain](https://python.langchain.com/)"
        )

        # PDF upload section
        uploaded_file = st.file_uploader("Upload a PDF (Optional)", type=["pdf"])
        #
        # st.divider()
        #
        # o = st.selectbox(label="Select a name space to delete",
        #                       options=("FannieMae.pdf","FreddieMac.pdf", "FHA", "VA"))
        # st.button("Delete Index", on_click=truncate_index)
    return uploaded_file


# Function to handle PDF upload and process the content
def handle_pdf_upload(uploaded_file):
    if uploaded_file:
        file_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split PDF content asynchronously
        with st.spinner("Processing PDF..."):
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            split_documents = text_splitter.split_documents(documents)
            vector_store.add_documents(split_documents)

def truncate_index():
    index.delete(namespace="FannieMae.pdf")

# Function to generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})


# Function to build the prompt chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(
                HumanMessagePromptTemplate.from_template(msg["content"])
            )
        elif msg["role"] == "ai":
            prompt_sequence.append(
                AIMessagePromptTemplate.from_template(msg["content"])
            )
    return ChatPromptTemplate.from_messages(prompt_sequence)


# Function to retrieve relevant documents from the vector store!!
def process_query_with_pdf(user_query):
    retrieved_docs = vector_store.similarity_search(user_query)
    print(retrieved_docs)
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"question": user_query, "context": context})
    return None


# function to handle the user query
def handle_user_query(user_query, uploaded_file):
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})

    ai_response = process_query_with_pdf(user_query)
    if ai_response is None:
        ai_response = model.invoke(user_query)

    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response.content})


# main Streamlit UI function
def main():
    apply_custom_css()
    uploaded_file = configure_sidebar()

    # Chat container on the main screen
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                print(message["content"])
                st.markdown(message["content"])

    # chat input and processing
    user_query = st.chat_input("Type your question here...")

    if uploaded_file:
        handle_pdf_upload(uploaded_file)
        st.rerun()

    # Handle user query
    if user_query:
        handle_user_query(user_query, uploaded_file)
        # rerun to update display
        st.rerun()


if __name__ == "__main__":
    main()