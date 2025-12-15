import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. 설정 및 API 키
# ==========================================

# 데이터 경로 (사용자 환경에 맞게 수정)
DATA_DIR = "C:/RAG_DATA/data/test"

# OpenRouter / OpenAI API 키 설정
# 실제 키를 여기에 입력하거나 환경 변수로 관리하세요.
# os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..." 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", YOUR_API_KEY)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

LLM_MODEL_NAME = "google/gemini-2.5-flash-preview-09-2025"
EMBEDDING_MODEL_NAME = "openai/text-embedding-3-small"

# ==========================================
# 2. 함수 정의 (모델 로드 및 답변 생성)
# ==========================================

@st.cache_resource
def load_faiss_index(folder_path, index_name):
    """
    FAISS 인덱스를 로드합니다. (캐싱 적용으로 속도 향상)
    """
    # [Fix] OpenRouter 사용 시 openai_api_base와 api_key를 명시적으로 전달해야 합니다.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_API_BASE,
        check_embedding_ctx_length=False
    )
    try:
        vectorstore = FAISS.load_local(
            folder_path=folder_path, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True 
        )
        return vectorstore
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

def generate_response(llm, context, question):
    """
    LLM을 사용하여 답변을 생성합니다.
    """
    prompt = f"""You are a helpful and rigorous AI assistant grounded in Database Systems knowledge.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()

# ==========================================
# 3. Streamlit UI 구성
# ==========================================

st.set_page_config(page_title="RAPTOR RAG Demo", page_icon="🦖", layout="wide")

st.title("🦖 RAPTOR RAG Q&A System")
st.markdown("데이터베이스 전공 서적 지식을 기반으로 한 **질의응답 시스템**입니다.")

# --- 사이드바: 모델 선택 ---
st.sidebar.header("⚙️ 설정 (Settings)")

# 모델 폴더 스캔
if os.path.exists(DATA_DIR):
    # .faiss 파일이 있는 폴더만 찾기 (재귀적 탐색)
    model_options = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".faiss"):
                # 경로에서 모델 이름 추출 (파일명 또는 폴더명)
                model_name = os.path.splitext(file)[0]
                full_path = root
                model_options.append((model_name, full_path))
    
    if not model_options:
        st.sidebar.warning(f"No .faiss files found in {DATA_DIR}")
        st.stop()

    # 선택 박스 (표시 이름: 모델명)
    selected_option = st.sidebar.selectbox(
        "사용할 모델 선택", 
        options=model_options, 
        format_func=lambda x: x[0] # 모델명만 표시
    )
    
    selected_model_name, selected_model_path = selected_option
    
    # 검색 설정
    top_k = st.sidebar.slider("검색할 문서 수 (Top-K)", min_value=1, max_value=10, value=5)

else:
    st.sidebar.error(f"경로를 찾을 수 없습니다: {DATA_DIR}")
    st.stop()

# --- 모델 로드 ---
if selected_model_name:
    vectorstore = load_faiss_index(selected_model_path, selected_model_name)
    if vectorstore:
        st.sidebar.success(f"✅ 모델 로드 완료: {selected_model_name}")
    else:
        st.stop()

# --- 메인 채팅 인터페이스 ---

# 세션 상태 초기화 (채팅 기록 저장)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 만약 이전에 검색된 문서가 있다면 표시 (assistant 메시지인 경우)
        if "docs" in message:
            with st.expander("📚 참고한 문서 (Retrieved Context) 확인하기"):
                for i, doc in enumerate(message["docs"]):
                    st.markdown(f"**[Document {i+1}]**")
                    st.text(doc.page_content)
                    st.divider()

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (예: What is ACID property?)"):
    # 1. 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 답변 생성 과정
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # (1) 검색 (Retrieval)
        with st.status("🔍 관련 문서를 검색하고 있습니다...", expanded=True) as status:
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                retrieved_docs = retriever.invoke(prompt)
                status.update(label="검색 완료!", state="complete", expanded=False)
                
                # 검색된 문서 미리보기 (Expander)
                with st.expander("📚 참고한 문서 (Retrieved Context) 확인하기"):
                    context_text = ""
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**[Document {i+1}]**")
                        st.text(doc.page_content)
                        st.divider()
                        context_text += doc.page_content + "\n\n"
            except Exception as e:
                status.update(label="검색 실패", state="error", expanded=True)
                st.error(f"검색 중 오류 발생: {e}")
                st.stop()

        # (2) 생성 (Generation)
        message_placeholder.markdown("🤖 답변을 생성 중입니다...")
        
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_API_BASE,
                temperature=0
            )
            
            answer = generate_response(llm, context_text, prompt)
            
            # (3) 결과 표시
            message_placeholder.markdown(answer)
            
            # 세션에 저장 (문서 정보 포함)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "docs": retrieved_docs
            })
        except Exception as e:
            st.error(f"답변 생성 중 오류 발생: {e}")