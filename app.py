import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# ===========================
# 🔐 Set Environment Tokens
# ===========================
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["HF_TOKEN"] = "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"

# ===========================
# 📂 Paths
# ===========================
DB_PATH = "/Users/rajbunny/rag_sumedha/db1"
TXT_FILE = "/Users/rajbunny/rag_sumedha/sumedha_rag.txt"

# ===========================
# 🎨 Streamlit UI Setup
# ===========================
st.set_page_config(page_title="📘 SystemVerilog Knowledge Base", layout="wide")
st.title("🧠 Query SystemVerilog Concepts (RAG Demo)")

# ===========================
# 🔍 Load Embedding Model
# ===========================
emb = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# ===========================
# 📄 Load & Split Document
# ===========================
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

# ===========================
# 🧱 Create / Load Chroma DB
# ===========================
if "vectordb" not in st.session_state:
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
        st.info("📄 Creating and embedding new Chroma database...")


        vectordb = Chroma.from_texts(
            texts=docs,embedding=emb,collection_name="verilog_db_2_0",ersist_directory=DB_PATH)
        st.success(f"✅ Created and automatically persisted Chroma DB with {len(docs)} chunks.")

    else:
        st.info("📦 Loading existing Chroma DB...")
        vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=emb,
            collection_name="verilog_db_2_0"
        )

    st.session_state.vectordb = vectordb
else:
    vectordb = st.session_state.vectordb

# ===========================
# 🔎 Create Retriever
# ===========================
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.75}
)

# ===========================
# 💬 User Query
# ===========================
query = st.text_input("🔍 Enter your question:")
strr = ""

if st.button("Submit Query"):
    if query:
        results = retriever.invoke(query)
        for i in results:
            strr += i.page_content
        st.success("✅ Retrieved relevant document context.")
        st.expander("📄 View Retrieved Context").write(strr[:2000] + "...")
    else:
        st.warning("Please enter a question first.")

# ===========================
# 🧠 Prompt Setup
# ===========================
system_prompt = SystemMessagePromptTemplate.from_template("""You are a SystemVerilog expert and documentation analyst with over 20 years of experience in verification and digital design.

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
- Reconstruct SystemVerilog code only when context describes it.
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

# ===========================
# 📦 Define Response Schema
# ===========================
class VerilogAnswer(BaseModel):
    answer_summary: str = Field(..., description="A short 1-2 line summary of the answer.")
    detailed_explanation: str = Field(..., description="Detailed explanation based only on the document context.")
    code_example: str = Field(..., description="SystemVerilog code example if present in the document context.")

jsp = PydanticOutputParser(pydantic_object=VerilogAnswer)

formatted_prompt = prompt.format(
    context=strr,
    question=query,
    fi=jsp.get_format_instructions()
)

# ===========================
# 🤖 HuggingFace LLM
# ===========================
deepseek = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.5,
    max_new_tokens=256,
    task="conversational",
    provider="sambanova"
)

deep_seek = ChatHuggingFace(
    llm=deepseek,
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.5,
    max_new_tokens=256,
    task="conversational",
    provider="sambanova",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# ===========================
# 🧾 Generate Response
# ===========================
if query and strr:
    response = deep_seek.invoke(formatted_prompt)
    st.subheader("💡 Response:")
    response_text = response.content


    pattern = re.compile(
    r'\{\s*"answer_summary"\s*:\s*".*?"\s*,\s*"detailed_explanation"\s*:\s*".*?"\s*,\s*"code_example"\s*:\s*".*?"\s*\}',
    re.DOTALL )

    m = pattern.search(response_text)
    scrapped = m.group(0) if m else None
    parsed_output = jsp.parse(scrapped)
    st.write(parsed_output.answer_summary)
    st.write(parsed_output.detailed_explanation)
    st.code(parsed_output.code_example,language="systemverilog")

