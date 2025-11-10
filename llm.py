from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-index"
    db = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_llm(model="gpt-5-mini"):
    return ChatOpenAI(model=model, temperature=0)


def get_dictionary_chain():
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

    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()

    client = Client()
    rag_prompt = client.pull_prompt("rlm/rag-prompt")

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_ai_message(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"question": dictionary_chain} | rag_chain
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message
