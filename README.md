# MedRAG — 以病患为中心的 AI 医生排班问答助手

MedRAG是一个基于 RAG 的AI 医生排班助手，能够智能回答对医生排班、就诊信息当等问题，并能显示推理过程，帮助患者做出更好决策。

## 一、安装与运行

### 依赖环境

- Python >= 3.9
- pip (Python package manager)

### 安装步骤

```shell
# 1. 先克隆项目
git clone git@github.com:shaynewx/MedRAG.git
cd MedRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 在根目录下新建.env，配置环境
DEEPSEEK_API_KEY="<your deepseek url>"

# 4. 运行应用(会自动跳转至http://localhost:8501)
streamlit run app.py
```



## 二、项目结构

```
MedRAG/
├── data/
│   ├── doctor_schedule.csv      # 医生排班原始数据
│   └── faiss_index/             # FAISS 向量索引
├── .env                         # 环境变量（API key ）
├── app.py                       # 主应用代码
├── build_index.py               # 构建 FAISS 索引
├── requirements.txt             # Python 依赖
└── README.md
```



## 三、技术架构 & 设计考量

### 基础技术

- **LangChain**: 构建 RAG 模型进行输入编排 + 返回输出
- **Moka-ai/m3e-base model**: 用于医生排班信息向量化
- **FAISS**: 优化的向量搜索库，高效量检索
- **DeepSeek API**: 实现 LLM 生成
- **Streamlit**: 快速打造可交互前端 UI

### 性能 & 可扩展性考量

| 组件     | 技术选型              | 选型原因                       |
| -------- | --------------------- | ------------------------------ |
| UI 展示  | Streamlit             | 快速上线前端                   |
| 向量搜索 | m3e-base mode + FAISS | 中文数据库友好，支持本地检索   |
| LLM API  | DeepSeek              | 支持中文，API费用相对便宜      |
| 历史存储 | JSON 本地文件         | 方便进行多会话管理，无需数据库 |



## 四、如果有更多时间，优先考虑的改进

1. **完善数据库，根据更复杂的数据库重新优化chunk方式**

2. **支持用户登录注册**

3. **完善多会话管理**

   - 如支持对话删除 / 重命名 
   - 用户可选择重新生成回答
   - 对话应倒序排序

4. **重构向量检索策略**

   - 加入稀疏检索与rerank
   - 二次检索时加入此前对话中的关键信息

5. **数据库化历史存储**

   - 将 `chat_history.json` 改为 SQLite 或 Supabase
   - 支持储存多用户数据

   

6. **寻找免费的LLM API**

7. **封装成小程序或应用等其他形式**

8. **从问答助手逐渐升级为Agent**



## 五、测试指南

无需特殊测试架，运行后可自然与 UI 交互，可测试指令：

- 王医生在吗？
- 我今天心脏不太舒服

