import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# 1. Setup Models
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


def setup_vector_store(pdf_path, store_name="faiss_index"):
    # Check if the local folder already exists
    if os.path.exists(store_name):
        print(f"Loading existing index: {store_name}")
        vectorstore = FAISS.load_local(
            store_name, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Index not found. Processing PDF...")
        # Your original setup logic
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Create and Save
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(store_name)
        print(f"Index saved to {store_name}")

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


# Initialize retriever (provide your PDF path here)
retriever = setup_vector_store("standard_5.pdf")


# 3. Define Tools
@tool
def search_pdf(query: str) -> str:
    """Search the uploaded PDF for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


tools = [search_pdf]

# 4. Agent Setup with Self-Reflection Logic
system_prompt = """You are a helpful AI Assistant with access to a PDF document.
Follow this process:
1. Search: Use the search_pdf tool to find information.
2. Reflect: Look at the results. If the information is missing or incomplete, try a different search query.
3. Answer: Provide a final answer only when you are confident. If the info isn't there, say so."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 5. Command Line Chat Loop
def start_chat():
    print("--- PDF RAG Agent Active (Type 'exit' to quit) ---")
    chat_history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history}
        )

        print(f"\nAI: {response['output']}")
        chat_history.append(("human", user_input))
        chat_history.append(("ai", response["output"]))


if __name__ == "__main__":
    start_chat()
