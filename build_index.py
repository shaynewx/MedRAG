import pandas as pd
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_data():
    """
    加载数据集并填充NaN值
    """

    df = pd.read_csv("data/doctor_schedule.csv")
    df = df.fillna("")
    return df


def df_to_chunks(df):
    """
    chunk数据集
    """

    chunks = []
    for _, row in df.iterrows():
        content = f"""医生：{row['doctor_name']}；科室：{row['department']}；职称：{row['title']}；时间：{row['date']} {row['start_time']} - {row['end_time']}；地点：{row['location']}；备注：{row['notes']}"""
        chunks.append(
            Document(page_content=content, metadata={"doctor_id": row["doctor_id"]})
        )
    return chunks


def build_and_save_faiss(chunks):
    """
    为数据集构建FAISS向量索引并保存
    """

    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("data/faiss_index")
    print("向量索引构建完成并保存至data/faiss_index")


if __name__ == "__main__":
    # 加载数据，分块、构建索引并保存
    df = load_data()
    chunks = df_to_chunks(df)
    print(chunks)
    build_and_save_faiss(chunks)
