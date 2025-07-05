import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import pytz

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
        st.session_state["history"] = []

    # 输入框交互
    user_input = st.chat_input("请输入你的问题...")

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

        # 显示检索片段
        with st.chat_message("assistant"):
            st.markdown("📚 **检索到的内容片段：**")
            for i, doc in enumerate(dense_results):
                st.markdown(f"**片段 {i+1}:**")
                st.code(doc.page_content, language="markdown")

            st.markdown("**正在接入大模型语言推理中，请稍候...**")

        # 构造 LLM 输入（历史记录+用户输入），并进行推理
        inputs = {"history": st.session_state["history"], "input": full_input}
        answer = (prompt | llm | output_parser).invoke(inputs)

        # 保存对话历史，并在下一次聊天中并入 prompt
        st.session_state["history"].append({"role": "user", "content": user_input})
        st.session_state["history"].append({"role": "assistant", "content": answer})

        # 展示对话历史（倒序渲染）
        for msg in st.session_state["history"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

        # 推理过程展示
        if user_input:
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


# TODO: 永久储存历史会话；
if __name__ == "__main__":
    main()
