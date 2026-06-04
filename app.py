import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Muat API Key
load_dotenv()

# --- UPGRADE 2: UI/UX BRANDING (Tampilan Depan & Sidebar) ---
st.set_page_config(page_title="Venue CS Assistant", page_icon="🍸", layout="centered")

# Membuat Sidebar (Menu Samping)
with st.sidebar:
    st.markdown("### 🍸 Venue CS Panel")
    st.markdown("Pusat bantuan reservasi cerdas yang didukung AI.")
    st.divider()
    st.markdown("📍 **Lokasi:** Jakarta, Indonesia")
    st.markdown("🕒 **Jam Operasional:** 18:00 - 04:00 WIB")
    st.divider()
    st.caption("Admin Tools:")
    # Tombol untuk mereset memori chat
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# Judul Utama
st.title("Venue Customer Service AI 🍸")
st.caption("Asisten pintar untuk menjawab pertanyaan reservasi dan aturan venue.")

@st.cache_resource
def init_rag_pipeline():
    # Baca dokumen dan ubah ke vektor (Database)
    loader = TextLoader("venue_knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Otak LLM (Menggunakan model terbaru Llama 3.1)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    
    # --- UPGRADE 1: CONVERSATION MEMORY (Daya Ingat AI) ---
    # A. Instruksi khusus agar AI mengingat konteks pertanyaan sebelumnya
    contextualize_q_system_prompt = (
        "Berdasarkan riwayat obrolan di bawah ini dan pertanyaan terbaru pengguna, "
        "susun ulang pertanyaan pengguna menjadi pertanyaan mandiri yang utuh "
        "tanpa perlu membaca riwayat. Jangan jawab pertanyaannya, cukup susun ulang."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # B. Instruksi Utama (Persona CS dengan Aturan Kasir Baru)
    qa_system_prompt = (
        "Anda adalah asisten customer service virtual premium untuk sebuah tempat hiburan malam. "
        "Gunakan informasi berikut untuk menjawab pertanyaan pengunjung. "
        "Jawablah dengan bahasa Indonesia yang ramah, asik, elegan, dan sopan. "
        "Sebutkan harga dengan format Rupiah (Rp) yang rapi. "
        "ATURAN PENTING: \n"
        "1. Jika aturan menyebutkan wajib DP 50%, berarti pengguna BOLEH dan SANGAT DIIZINKAN untuk membayar lunas (100%) di awal. Jangan pernah menolak pelanggan yang ingin membayar lunas.\n"
        "2. Anda adalah bot informasi, bukan kasir. Jika pengguna setuju untuk memesan atau membayar, berikan total harganya lalu ARAHKAN mereka untuk menghubungi WhatsApp Admin (0811-2233-4455) untuk proses transfer pembayaran.\n"
        "Jika ada pertanyaan di luar konteks tempat hiburan, tolak dengan sangat sopan.\n\n"
        "Konteks Venue:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Menggabungkan komponen menjadi satu pipeline utuh
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Inisialisasi Bot
chain = init_rag_pipeline()

# --- PESAN SAMBUTAN OTOMATIS ---
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Selamat datang di layanan reservasi kami. Ada yang bisa saya bantu untuk rencana malam Anda? ✨"}
    ]

# Tampilkan riwayat chat di layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom Input Pengguna
if user_input := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Simpan dan tampilkan chat pengguna
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- MEMPROSES MEMORI UNTUK LLM ---
    chat_history = []
    for msg in st.session_state.messages[:-1]: # Ambil semua riwayat kecuali input terakhir
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    # Menghasilkan dan menampilkan respons AI
    with st.chat_message("assistant"):
        with st.spinner("Mengecek informasi..."):
            response = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            
    # Simpan jawaban AI ke dalam riwayat
    st.session_state.messages.append({"role": "assistant", "content": answer})
