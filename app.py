import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
import tempfile
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Medical PDF Assistant",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def process_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Initialize the BAAI embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store using FAISS
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

def create_conversation_chain(vectorstore):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit"
    )
    # Load PEFT model
    model = PeftModel.from_pretrained(
        base_model, "saisuryateja1436/medical-llama3.2-1b-sft"
    )
    # Create a text generation pipeline
    pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        
          
    )
    # Wrap with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main application interface
st.title("🏥 Medical PDF Assistant")
st.subheader("Your AI-powered medical document analysis companion")

# Upload PDF section
with st.container():
    st.markdown("### 📄 Upload Medical Documents")
    pdf_docs = st.file_uploader(
        "Upload your medical PDFs here",
        type=['pdf'],
        accept_multiple_files=True
    )

    if pdf_docs:
        if st.button("Process Documents"):
            with st.spinner("Processing your medical documents..."):
                # Get PDF text
                raw_text = process_pdf(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)
                
                st.session_state.processing_complete = True
                st.success("Documents processed successfully!")

# Chat interface
if st.session_state.processing_complete:
    st.markdown("### 💬 Chat with your Medical Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, dict):
            role = "Assistant" if message.get("type") == "assistant" else "You"
            content = message.get("content", "")
            with st.chat_message(role.lower()):
                st.write(content)

    # Chat input
    user_question = st.chat_input("Ask a question about your medical documents...")
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.conversation is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({
                        'question': user_question,
                        'chat_history': st.session_state.chat_history
                    })
                    
                    st.write(response['answer'])
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "type": "human",
                        "content": user_question
                    })
                    st.session_state.chat_history.append({
                        "type": "assistant",
                        "content": response['answer']
                    })
        else:
            st.error("Please process documents before starting a conversation.")

# Add helpful instructions
with st.sidebar:
    st.markdown("### 📌 How to Use")
    st.markdown("""
    1. Upload one or more medical PDF documents
    2. Click 'Process Documents' to analyze them
    3. Ask questions about the medical content
    4. Get AI-powered responses based on your documents
    
    ### 🔍 Features
    - Multi-document support
    - Advanced medical language model
    - Contextual understanding
    - Conversation memory
    
    ### ⚠️ Disclaimer
    This tool is for informational purposes only and should not replace professional medical advice.
    """) 