import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import langchain

langchain.debug_mode = True
langchain.debug = True

# Setup
openai_api_key = "sk-proj-sk-proj-addkey"
persist_directory = "/vectordb/vectordb_files"  # directory to store the vector database files

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)



retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)

# Prompt Templates
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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise. Use ten sentences maximum if the task is a summary."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# Summarization Agent Setup
def summarize_document(llm, vectordb, question):
    # Retrieve the most relevant document using similarity
    docs = vectordb.similarity_search(question, k=1)
    document_title = docs[0].metadata['title']
    filtered_docs = vectordb.similarity_search(query="", k=30, filter={"title": document_title})
    
    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
 
    return stuff_chain.invoke(filtered_docs)["output_text"], document_title

def is_summarization_request(question):
    # Detect various forms of the word "summary" or "summarize"
    keywords = ["summary", "summarize", "summarizing"]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in keywords)

# Streamlit App
st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if is_summarization_request(prompt):
        summary, doc_title = summarize_document(llm, vectordb, prompt)
        response = f"**Summary:**\n{summary}\n\n**Source Document:**\n{doc_title}"
    else:
        input_data = {"input": prompt}
        config = {"configurable": {"session_id": "1"}}
        result = conversational_rag_chain.invoke(input_data, config=config)
        response = result['answer']
        # Display source documents and chuck used for the answer
        response += "\n\n\n\n**Source Documents:**\n"
        for i, doc in enumerate(result['context']):
            response += f"\nDocument {i + 1}:\nMetadata: {doc.metadata}\n\n**Text**: {doc.page_content[:400]}\n"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
