from dotenv import load_dotenv
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0.2,)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

character = "the main character"

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "The retrieved context is a book."
    f"Act as {character} from the book."
    "Try to use his/hers vocabulary."
    "Use the following pieces of retrieved context to answer the questions."
    "If you don't know the answer, say that you don't know."
    "Try to keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ],
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt,)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="answer",
    history_messages_key="chat_history",
)

while True:
    content = input(">>>")
    if content == 'q':
        break
    result = chain_with_message_history.invoke({"input": content},
                                               {"configurable": {"session_id": "unused"}}, )
    print(result['answer'])
