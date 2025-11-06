from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_ai_message(user_message):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-index"
    db = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    client = Client()
    rag_prompt = client.pull_prompt("rlm/rag-prompt")
    retriever = db.as_retriever(search_kwargs={"k": 4})

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

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

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    tax_chain = {"question": dictionary_chain} | rag_chain
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message


st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세 관련 질문을 입력하세요.")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세에 관련된 질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성중입니다..."):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
