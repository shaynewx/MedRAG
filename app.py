import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import pytz
import json

# 用文件保存历史记录
HISTORY_FILE = "chat_history.json"


# 关闭 huggingface 的 加速tokenizers编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.cache_resource
def load_vectorstore():
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # 显示流式内容+光标

# 加载历史
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 保存历史
def save_chat_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    # 初始化前端环境
    st.set_page_config(page_title="医疗助手RAG", layout="wide")
    st.title("医疗系统问答助手 ")
    st.write("请输入你的问题，支持多轮追问。")

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

    output_parser = StrOutputParser()  # 把生成结果解析为纯文本字符串

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是服务于病患的医疗助手，请温柔耐心、依据事实以及历史对话回答用户问题，如果用户提示不清晰，请猜测用户接下来想说的内容并给到用户引导。",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "user",
                "{input}\n请详细说明你是如何得出答案的，展示你的思考过程。如果检索内容无关，请说明理由。",
            ),
        ]
    )

    # 构造 Streamlit 聊天 以及 对话历史
    if "history" not in st.session_state:
        st.session_state["history"] = load_chat_history()

    # 输入框交互
    user_input = st.chat_input("请输入你的问题...")

    # 展示所有历史消息
    for msg in st.session_state["history"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    if user_input:
        # 展示用户提问
        st.chat_message("user").write(user_input)

        # 获取当前时间信息
        beijing_full_time, time_period = get_beijing_time()

        # 进行稠密检索 chunk
        dense_results = vectorstore.similarity_search(user_input, k=8)
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
            chain = LLMChain(prompt=prompt, llm=llm)
            inputs = {"history": st.session_state["history"], "input": full_input}
            answer = chain.invoke(inputs)["text"]

        # 只 append assistant 输出
        st.session_state["history"].append({"role": "assistant", "content": answer})

        # 推理过程展示
        with st.expander("模型推理过程（点击展开）"):
            st.markdown("**原始用户问题：**")
            st.code(user_input, language="markdown")

            st.markdown("**检索到的内容片段：**")
            for i, doc in enumerate(dense_results):
                st.markdown(f"**片段 {i+1}:**")
                st.code(doc.page_content, language="markdown")

            st.markdown("**传给模型的完整 Prompt 输入：**")
            st.code(full_input, language="markdown")

            st.markdown("**最终回答：**")
            st.code(answer, language="markdown")


# TODO: 当前第二次提问后前一次推理过程被覆盖无法查看；检索可加入稀疏检索然后进行多路召回
if __name__ == "__main__":
    main()
