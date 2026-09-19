import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

def main():
    # -------------------------
    # Initialize session_state BEFORE creating widgets
    # -------------------------
    if 'button_query' not in st.session_state:
        st.session_state['button_query'] = ""
    if 'run_button_query' not in st.session_state:
        st.session_state['run_button_query'] = False

    
    # ----------------------------------------------
    with st.sidebar:
        st.title('CHAT WITH PDF📚')

        st.markdown('''
        ## About
        This is an AI powered PDF chatbot that answers your query using only your uploaded PDF.
        Upload a PDF and ask your query.
        ''')
        add_vertical_space(3)
        
        # Model Info / Settings
        st.markdown("### ⚙️ Model Info")
        st.write("- FLAN-T5 Base (Local)")
        st.write("- Embeddings: MiniLM L6-v2")
        add_vertical_space(2)

        # About / Credits
        st.markdown("### 🦋 Credits")
        st.write("Made by Sanskriti")

    # ----------------------------------------------
    # Main App
    # ----------------------------------------------
    st.header("CHAT WITH YOUR PDF💬")

    pdf = st.file_uploader("Upload your PDF here", type="pdf")

    if pdf is not None:
        # Extract text
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Load / create embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("✅ Embeddings loaded from file.")
        else:
            st.write("🔍 Creating new embeddings (may take a moment)...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write("✅ Embeddings created and saved.")

        # Load local model
        model_path = "./models/flan-t5-base"
        if not os.path.exists(model_path):
            st.error("❌ Model not found!.")
            st.stop()

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_length=512, 
            device = -1   
 
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=VectorStore.as_retriever(), 
            chain_type="stuff" 
        )

        # ----------------------------------------------
        # Buttons for predefined actions
        # ----------------------------------------------
        st.markdown("""
                 <style>
                 div.block-container {
                 padding-top: 1rem;
                 padding-bottom: 1rem;
                                    }
                </style>
                  """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3) 
        with col1:
            if st.button("Summarize", key="btn_summarize"):
                st.session_state['button_query'] = "Summarize this PDF"
                st.session_state['run_button_query'] = True 
        with col2:  
            if st.button("Extract Keywords", key="btn_keywords"):
                st.session_state['button_query'] = "Extract important keywords from this PDF"
                st.session_state['run_button_query'] = True
        with col3:
            if st.button("Generate Q&A", key="btn_qa"):
                st.session_state['button_query'] = "Generate possible questions and answers from this PDF"
                st.session_state['run_button_query'] = True

        # ----------------------------------------------
        # User Question Input
        # ----------------------------------------------
        query_input = st.text_input(
            "Ask a question about your PDF:",
            value="",
            key="query"
        )

        # ----------------------------------------------
        # Determine query to run
        # ----------------------------------------------
        query_to_run = ""
        if st.session_state.get('run_button_query', False):
            query_to_run = st.session_state['button_query']
            st.session_state['run_button_query'] = False
        elif query_input:
            query_to_run = query_input

        # ----------------------------------------------
        # Run QA
        # ----------------------------------------------
        if query_to_run:
            with st.spinner("Thinking..."):
                answer = qa.run(query_to_run)
            st.success("✅ Answer:")
            st.write(answer)

        # ----------------------------------------------
        # Tip
        # ----------------------------------------------
        st.markdown(
            """
            <style>
            .suggestion-tip {
                text-align:center;
                margin-top: 18px;
                color: #1F456E;
                font-size:15px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='suggestion-tip'>💡 Tip: Try asking 'Summarize this PDF' or 'Find key points.'</div>",
            unsafe_allow_html=True
        )

    else:
        st.info("📄 Please upload a PDF to start chatting.")

    
    # ----------------------------------------------
    # Sidebar color + rounded edges
    # ----------------------------------------------
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #4F76AF;
            border-radius: 25px;  /* Rounded corners */
            margin: 10px;
            padding: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        [data-testid="stSidebar"] > div:first-child {
            border-radius: 25px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Main background gradient (with white fix)
    # ----------------------------------------------
    st.markdown(
        """
        <style>
        /* Full-page gradient */
        .stApp {
            background: linear-gradient(135deg, #BAD6F2 0%, #E4EEF8 100%) !important;
            height: 100vh;
            margin: 0;
            padding: 0;
        }

        /* Remove top white bar */
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Ensure all containers blend */
        [data-testid="stToolbar"], .st-emotion-cache-1dp5vir, .block-container {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Button styling
    # ----------------------------------------------
   
    st.markdown(
        """
        <style>
        div.stButton > button { 
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            border: none;
            transition: 0.3s;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            background-color: #5A9CF2;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Add subtle shadow to main card / container
    # ----------------------------------------------
    st.markdown(
        """
        <style>
        [data-testid="stVerticalBlock"] {
            background-color: rgba(255, 255, 255, 0.65);
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Footer: "Made by Sanskriti 💡"
    # ----------------------------------------------
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            margin-top: 40px;
            transition: all 0.3s ease;
        }
        .footer:hover {
            color: #4A90E2;
            text-shadow: 0 0 8px rgba(74,144,226,0.6);
        }
        </style>
        <div class="footer">Made by Sanskriti 💡</div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
