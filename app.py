import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import re

# ===========================
# 🔐 Load Environment Tokens
# ===========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API
os.environ["HF_TOKEN"] = HF_TOKEN

# ===========================
# 🎨 Streamlit UI Setup
# ===========================
st.set_page_config(page_title="📘 SystemVerilog Knowledge Base", layout="wide")

# --- Your CSS ---
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 38px;
            color: #00A8E8;
            font-weight: 800;
            margin-bottom: -5px;
        }
        .sub-text {
            text-align: center;
            font-size: 16px;
            color: #5f6368;
            margin-top: 0;
            margin-bottom: 25px;
        }
        .stChatInputContainer textarea {
            border: 2px solid #00A8E8 !important;
            border-radius: 12px !important;
            font-size: 18px !important;
            padding: 12px !important;
            min-height: 60px !important;
        }
        .stChatMessage.user {
            background-color: #e3f6ff !important;
            border-left: 5px solid #00A8E8 !important;
        }
        .stChatMessage.assistant {
            background-color: #f1f9ff !important;
            border-left: 5px solid #0077b6 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Titles
st.markdown('<div class="main-title">🧠 SystemVerilog RAG Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Ask any question strictly from the SystemVerilog document</div>', unsafe_allow_html=True)

# ===========================
# Chat Memory
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ===========================
# Embedding Model
# ===========================
# ===========================
# Embedding Model
# ===========================
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ===========================
# Document + DB
# ===========================
DB_PATH = "db1"
TXT_FILE = "sumedha_rag.txt"

# 🔥 FIX: Create DB folder if missing (Streamlit Cloud needs this)
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Load document
with open(TXT_FILE, "r", encoding="utf-8") as f:
    text_data = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=[
        "\nCHAPTER ", "\nSection ", "\n\n", "\n• ", "\n- ",
        "\nExample", "\nFigure", "\nTable", "\n", ". ", " "
    ]
)

docs = splitter.split_text(text_data)
st.write(f"✅ Total chunks created: {len(docs)}")

# Create / load DB
if "vectordb" not in st.session_state:
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
        vectordb = Chroma.from_texts(
            texts=docs,
            embedding=emb,
            collection_name="verilog_db_2_0",
            persist_directory=DB_PATH
        )
    else:
        vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=emb,
            collection_name="verilog_db_2_0"
        )
    st.session_state.vectordb = vectordb
else:
    vectordb = st.session_state.vectordb

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.75}
)

# ===========================
# Chat Input
# ===========================
query = st.chat_input("Ask your SystemVerilog question...")

if query:

    # Show user msg
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # Retrieval
    strr = ""
    results = retriever.invoke(query)
    for r in results:
        strr += r.page_content

    # ===========================
    # YOUR SYSTEM MESSAGE (UNCHANGED)
    # ===========================
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a SystemVerilog expert and documentation analyst with over 20 years of experience in verification and digital design.

Your task is to answer questions strictly and only using the retrieved document context.  
You must maintain multi-turn continuity, but document context always overrides conversation memory.

==================== RULES FOR USING DOCUMENT CONTEXT ====================

1. Always use the retrieved document context.
2. If the context contains partial information, you may logically complete it, but stay faithful.
3. Never ignore the context because of different wording.
4. Never introduce information outside the document.
5. If the document contains no relevant information, respond exactly with:
The document does not contain enough information to answer this question.

==================== MULTI-TURN RULES ====================

1. Use previous user messages and your previous answers to maintain continuity.
2. Do not repeat long explanations unless asked.
3. Never override document context with conversation memory.

==================== OUTPUT FORMAT RULES ====================

You MUST always output a single valid JSON object ONLY.

The JSON must follow EXACTLY this schema:

{{
  "answer_summary": "A short summary of the answer based on the document context.",
  "detailed_explanation": "A complete explanation using information and reasoning supported by the document context.",
  "code_example": "SystemVerilog code example reconstructed or quoted from the document if applicable."
}}

==================== EXTRA RULES ====================

- No markdown.
- No backticks.
- No <think>.
- No commentary outside the JSON.
- You MUST always generate a SystemVerilog code example whenever possible based on the concepts in the context, even if the exact code is not shown. Use minimal faithful reconstruction. or else say code is not required
- Always return valid JSON.
""")

    human_prompt = HumanMessagePromptTemplate.from_template("""
Answer the following question using only the provided SystemVerilog document context below.

Document Context:
{context}

Question:
{question}
""")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    class VerilogAnswer(BaseModel):
        answer_summary: str
        detailed_explanation: str
        code_example: str

    jsp = PydanticOutputParser(pydantic_object=VerilogAnswer)

    formatted_prompt = prompt.format(
        context=strr,
        question=query,
        fi=jsp.get_format_instructions()
    )

    # ===========================
    # Gemini LLM
    # ===========================
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.4,
        max_output_tokens=2048
    )

    response = gemini.invoke(formatted_prompt)
    response_text = response.content

    pattern = re.compile(
        r'\{[\s\S]*?"answer_summary"[\s\S]*?"detailed_explanation"[\s\S]*?"code_example"[\s\S]*?\}',
        re.DOTALL
    )

    m = pattern.search(response_text)
    extracted_json = m.group(0) if m else None

    parsed_output = jsp.parse(extracted_json)

    answer_block = (
        parsed_output.answer_summary
        + "\n\n"
        + parsed_output.detailed_explanation
    )

    st.chat_message("assistant").write(answer_block)
    st.code(parsed_output.code_example, language="systemverilog")

    st.session_state.messages.append({"role": "assistant", "content": answer_block})
