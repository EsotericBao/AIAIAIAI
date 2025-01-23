from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from config import CHROMADB, COLLECTION, Models

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

# Initialize the vector store
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMADB,  # Where to save data locally
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Your name is CHAT, a helpful assistant.
            Always maintain professionalism and a concise tone in your responses. 

            Guidelines for your responses:
            - Base your answers solely on the retrieved context. If the context does not provide enough information, say: 
            "I couldn't find relevant information to answer your question based on the data provided."
            - Do not fabricate information or provide answers beyond the context.
            - Keep your responses short, direct, and relevant to the context.
            - Avoid repeating information unnecessarily or introducing yourself repeatedly.
            - Respond to normal conversation topics

            If you understand these instructions, proceed to answer the user's question.


    Your responsibility is to assist users by answering questions based strictly on the provided context.
         """),

        ("human", 
         """Use the following question and context to generate a response:
         Question: {input}
         Context: {context}
        Ensure your answer is based only on the data provided.
         """)
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 20})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Fallback response
fallback_response = "I'm sorry, I couldn't find relevant information to answer your question based on the provided data."

# Main loop
def main():
    print("Welcome to CHAT! Ask your question or type 'q', 'quit', or 'exit' to end the session.")
    print("Type 'clear' or 'reset' to start a new session.")
    
    # Initialize chat history
    history = []
    
    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ")
        if query.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        elif query.lower() in ['clear', 'reset']:
            # Clear chat history
            history = []
            print("Chat history cleared. You can start a new conversation.")
            continue
        
        try:
            # Retrieve context and answer query
            result = retrieval_chain.invoke({"input": query})
            answer = result.get("answer", fallback_response)
            if not answer.strip():
                answer = fallback_response
            
            # Store the conversation in history
            history.append({"user": query, "assistant": answer})
            
            print("CHAT: ", answer, "\n\n")
        except Exception as e:
            print("CHAT: I encountered an error. Please try again later.\n\n")
            print(f"Error: {e}")  # Debugging: print the error

# Run the main loop
if __name__ == "__main__":
    main()
