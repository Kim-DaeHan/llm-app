from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-index"
    db = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_llm(model="gpt-4o-mini"):
    return ChatOpenAI(model=model, temperature=0)


# State 정의
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: str
    answer: str


# Dictionary 변환 노드
def transform_query(state: State):
    """사용자 질문을 용어 사전 기반으로 변환"""
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    dictionary_prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}

        질문: {{question}}
    """
    )

    dictionary_chain = dictionary_prompt | llm | StrOutputParser()
    transformed_question = dictionary_chain.invoke({"question": state["question"]})

    return {"question": transformed_question}


# 문서 검색 노드
def retrieve_documents(state: State):
    """대화 히스토리를 고려하여 관련 문서 검색"""
    retriever = get_retriever()
    llm = get_llm()

    # 대화 히스토리를 고려한 독립적인 질문으로 재구성
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("messages"),
            ("human", "{question}"),
        ]
    )

    # 대화 히스토리가 있으면 질문 재구성
    if len(state["messages"]) > 0:
        contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
        standalone_question = contextualize_chain.invoke(
            {"messages": state["messages"], "question": state["question"]}
        )
    else:
        standalone_question = state["question"]

    # 문서 검색
    docs = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    return {"context": context}


# 답변 생성 노드
def generate_answer(state: State):
    """검색된 문서를 기반으로 답변 생성"""
    llm = get_llm()

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("messages"),
            ("human", "{question}"),
        ]
    )

    qa_chain = qa_prompt | llm | StrOutputParser()

    answer = qa_chain.invoke(
        {
            "context": state["context"],
            "messages": state["messages"],
            "question": state["question"],
        }
    )

    # 대화 히스토리에 질문과 답변 추가
    return {
        "answer": answer,
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
    }


# 스트리밍 답변 생성 함수
def generate_answer_stream(context: str, messages: list, question: str):
    """검색된 문서를 기반으로 답변을 스트리밍 생성"""
    llm = get_llm()

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("messages"),
            ("human", "{question}"),
        ]
    )

    qa_chain = qa_prompt | llm | StrOutputParser()

    # 스트리밍 실행
    for chunk in qa_chain.stream(
        {"context": context, "messages": messages, "question": question}
    ):
        yield chunk


# Graph 생성
def create_graph():
    workflow = StateGraph(State)

    # 노드 추가
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)

    # 엣지 연결
    workflow.add_edge(START, "transform_query")
    workflow.add_edge("transform_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")

    # 메모리 체크포인터 추가
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


def get_ai_response(user_message: str, thread_id: str = "hans123"):
    """사용자 메시지를 처리하고 AI 응답을 실시간 스트리밍"""
    graph = create_graph()

    config = {"configurable": {"thread_id": thread_id}}

    # 현재 상태 가져오기 (히스토리 포함)
    try:
        current_state = graph.get_state(config)
        messages = list(current_state.values.get("messages", []))
    except Exception:
        messages = []

    # Step 1: Dictionary 변환
    initial_state = {
        "messages": messages,
        "question": user_message,
        "context": "",
        "answer": "",
    }

    # transform_query와 retrieve_documents 노드만 실행하여 context 획득
    state_after_retrieval = None
    for state in graph.stream(initial_state, config, stream_mode="values"):
        if "context" in state and state["context"]:
            state_after_retrieval = state
            break

    # Step 2: 답변을 스트리밍으로 생성
    if state_after_retrieval:
        full_answer = ""
        for chunk in generate_answer_stream(
            context=state_after_retrieval["context"],
            messages=state_after_retrieval["messages"],
            question=state_after_retrieval["question"],
        ):
            full_answer += chunk
            yield chunk

        # Step 3: 대화 히스토리에 저장
        final_state = {
            **state_after_retrieval,
            "answer": full_answer,
            "messages": state_after_retrieval["messages"]
            + [HumanMessage(content=user_message), AIMessage(content=full_answer)],
        }

        # 상태 업데이트
        graph.update_state(config, final_state)
