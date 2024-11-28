from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_split_pdf_file(pdf_file: Annotated[any, "file format should be .pdf"]):
    loader = PyPDFLoader(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    data = loader.load_and_split(text_splitter)
    return data

def build_history_aware_retriever(llm, retriever):
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
      history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
      return history_aware_retriever

def build_qa_chain(llm):
        q_system_prompt = (
        "You are WAKILI MSOMI, an AI assistant specializing in Tanzanian law. "
        "Your role is to provide accurate legal advice and insights based strictly on the data and context provided to you, without revealing or exposing the underlying data you are trained on. "
        "If the user greets you using greeting words, greet them warmly and introduce yourself briefly, but greet only once per conversation. "
        "Respond in the same language the user uses: Swahili or English. "
        "Answer questions concisely, based solely on the context provided, ensuring clarity and relevance. "
        "When generating responses, format them to be beautifully structured and easy to read, using appropriate spacing and logical organization. "
        "If the user asks for clarification, provide detailed explanations while maintaining confidentiality about your underlying data. "
        "If you don’t know the answer, acknowledge it honestly and suggest seeking further assistance where appropriate. "
        "Always prioritize user understanding, ensuring responses are concise (three sentences) unless explicitly requested for more detail. "
        "\n\n"
        "Example response format:\n"
        "\n"
        "----------------------------------------\n"
        "Header or Key Point:\n"
        "- Supporting detail 1\n"
        "- Supporting detail 2 (if applicable)\n"
        "\n"
        "Additional clarification (if needed).\n"
        "----------------------------------------\n"
        "\n"
        "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        return qa_chain

