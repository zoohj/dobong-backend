import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")

# --- 벡터 저장소 ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if VECTOR_DB_URL:
    from langchain_postgres import PGVector

    ASYNC_VECTOR_DB_URL = VECTOR_DB_URL.replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    vectorstore = PGVector(
        connection=ASYNC_VECTOR_DB_URL,
        embeddings=embeddings,
        collection_name="ai_chatbot",
        async_mode=True,
    )
else:
    from langchain_chroma import Chroma

    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="ai_chatbot",
        persist_directory="./chroma_db",
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Tool ---
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="AI Brief 문서에서 AI 산업 동향, 기업 전략, 기술 트렌드 등을 검색한다.",
)

# --- 그래프 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

graph = None


async def init_graph():
    global graph
    from psycopg import AsyncConnection
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # conn = await AsyncConnection.connect(DATABASE_URL, autocommit=True)
    # checkpointer = AsyncPostgresSaver(conn)
    checkpointer = await AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    await checkpointer.setup()

    graph = create_agent(
        model=llm,
        tools=[retriever_tool],
        checkpointer=checkpointer,
    )
    return checkpointer
