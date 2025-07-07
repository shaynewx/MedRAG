import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import load_dotenv
from datetime import datetime
import pytz
import json

# 用文件保存历史记录
HISTORY_FILE = "chat_history.json"


# 关闭 huggingface 的 加速tokenizers编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.cache_resource
def load_vectorstore():
    """
    返回向量库, 包含chunks、对应向量、向量索引结构
    """
    embedding_model = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    return FAISS.load_local(
        "data/faiss_index", embedding_model, allow_dangerous_deserialization=True
    )


def get_beijing_time():
    """
    返回当前北京时间：日期 + 时间；日期；返回 AM 或 PM
    """
    tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M"), now.strftime("%p")


# 继承 langchain 提供的 BaseCallbackHandler，每生成一个token就触发一次回调
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # 显示流式内容+光标


# 加载所有历史记录
def load_all_histories():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 保存聊天记录
def save_all_histories(histories):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)


def main():
    # 初始化前端环境
    st.set_page_config(page_title="医疗助手RAG", layout="wide")
    st.title("医疗系统问答助手 ")
    st.write("请输入你的问题，支持多轮追问。")

    # 设置Sidebar用于管理多个对话
    st.sidebar.title("🧾 对话管理")

    # 加载全部历史
    all_histories = load_all_histories()

    # 获取所有对话 ID（默认标题显示前几句）
    chat_ids = list(all_histories.keys())
    chat_titles = [
        all_histories[cid][0]["content"][:20] if all_histories[cid] else cid
        for cid in chat_ids
    ]

    # 显示所有已有对话作为按钮
    for cid, title in zip(chat_ids, chat_titles):
        if st.sidebar.button(title, key=f"chat_btn_{cid}"):
            st.session_state["chat_id"] = cid
            st.session_state["history"] = all_histories.get(cid, [])
            st.rerun()

    # 新建对话按钮
    if st.sidebar.button("➕ 新建对话"):
        new_id = f"chat_{len(chat_ids)+1}"
        all_histories[new_id] = []
        save_all_histories(all_histories)
        st.session_state["chat_id"] = new_id
        st.session_state["history"] = []
        st.rerun()

    # 加载 FAISS
    vectorstore = load_vectorstore()
    load_dotenv()

    # 初始化 LLM 和 prompt
    llm = ChatOpenAI(
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是服务于病患的医疗助手，请温柔耐心、依据事实以及历史对话回答用户问题，如果用户提示不清晰，请猜测用户接下来想说的内容并给到用户引导。",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "user",
                "{input}\n请结合备注，详细说明你是如何得出jie'lun的，展示你的思考过程。如果检索内容无关，请说明理由。",
            ),
        ]
    )

    # 用户首次进入streamlit时，初始化 Streamlit 聊天 以及 对话历史
    if "chat_id" not in st.session_state:
        if chat_ids:
            st.session_state["chat_id"] = chat_ids[0]
            st.session_state["history"] = all_histories.get(chat_ids[0], [])
        else:
            st.session_state["chat_id"] = "chat_1"
            st.session_state["history"] = []

    # 输入框交互
    user_input = st.chat_input("请输入你的问题...")

    # 展示所有历史消息以及推理过程
    history = st.session_state["history"]
    for i, msg in enumerate(history):
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

            # 如果当前是最后一条 assistant 消息，可选择展示推理信息
            is_last = i == len(history) - 1
            is_assistant = msg["role"] == "assistant"
            if is_last and is_assistant and "last_answer_meta" in st.session_state:
                meta = st.session_state["last_answer_meta"]
                with st.expander("🧠 模型推理过程（点击展开）"):
                    st.markdown("**原始用户问题：**")
                    st.code(meta["user_input"], language="markdown")

                    st.markdown("**检索到的内容片段：**")
                    for j, chunk in enumerate(meta["dense_results"]):
                        st.markdown(f"**片段 {j+1}:**")
                        st.code(chunk, language="markdown")

                    st.markdown("**传给模型的完整 Prompt 输入：**")
                    st.code(meta["full_input"], language="markdown")

                    st.markdown("**最终回答：**")
                    st.code(meta["answer"], language="markdown")

                # 展示后立即清除
                del st.session_state["last_answer_meta"]

    if user_input:
        # 展示用户提问
        st.chat_message("user").write(user_input)

        # 获取当前时间信息
        beijing_full_time, time_period = get_beijing_time()

        # 进行稠密检索 chunk
        dense_results = vectorstore.similarity_search(user_input, k=10)
        context = "\n\n".join([doc.page_content for doc in dense_results])

        # 拼接 prompt 输入
        full_input = f"当前北京时间：{beijing_full_time, time_period}\n\n问题：{user_input}\n\n以下是可能相关的医生排班信息：\n{context}"

        # 只 append 最原始的用户输入到历史中
        st.session_state["history"].append({"role": "user", "content": user_input})

        # 创建 assistant 对话容器并流式输出
        with st.chat_message("assistant"):
            # 创建流式输出容器
            msg_container = st.empty()

            # 初始化流式 handler
            stream_handler = StreamHandler(msg_container)
            llm.callbacks = [stream_handler]

            # 构造 LLM 输入（历史记录+本轮完整上下文），并进行推理
            chain = prompt | llm
            inputs = {"history": st.session_state["history"], "input": full_input}
            answer = chain.invoke(inputs).content

            # 储存推理细节（只能一轮展示）
            st.session_state["last_answer_meta"] = {
                "user_input": user_input,
                "dense_results": [doc.page_content for doc in dense_results],
                "full_input": full_input,
                "answer": answer,
            }

        # 添加回答
        st.session_state["history"].append({"role": "assistant", "content": answer})

        # 保存到多会话总记录中
        all_histories[st.session_state["chat_id"]] = st.session_state["history"]
        save_all_histories(all_histories)

        # 刷新，显示会话信息
        st.rerun()


if __name__ == "__main__":
    main()
