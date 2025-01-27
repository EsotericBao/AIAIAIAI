from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from config import CHROMADB, COLLECTION, Models

class Chatbot:
    def __init__(self):
        """
        Initialize the chatbot with models, vector store, and prompt template.
        """
        self.models = Models()
        self.embeddings = self.models.embeddings_ollama
        self.llm = self.models.model_ollama

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMADB,
        )

        # Define prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 """Your name is CHAT, a helpful assistant.
                    Always maintain professionalism and a concise tone in your responses. 

                    Guidelines for your responses:
                    - Base your answers solely on the retrieved context. If the context does not provide enough information, say: 
                    \"I couldn't find relevant information to answer your question based on the data provided.\"
                    - Do not fabricate information or provide answers beyond the context.
                    - Keep your responses short, direct, and relevant to the context.
                    - Avoid repeating information unnecessarily or introducing yourself repeatedly.
                    - Respond to normal conversation topics

                    If you understand these instructions, proceed to answer the user's question.
                """),

                ("human", 
                 """Use the following question and context to generate a response:
                 Question: {input}
                 Context: {context}
                 Ensure your answer is based only on the data provided.
                 """)
            ]
        )

        # Create retrieval chain
        self.retriever = self.vector_store.as_retriever(kwargs={"k": 20})
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)

        # Fallback response
        self.fallback_response = "I'm sorry, I couldn't find relevant information to answer your question based on the provided data."

    def get_response(self, query):
        """
        Get a response from the chatbot based on the input query.

        Args:
            query (str): The user's question.

        Returns:
            str: The chatbot's response.
        """
        try:
            # Retrieve context and generate response
            result = self.retrieval_chain.invoke({"input": query})
            answer = result.get("answer", self.fallback_response)
            return answer if answer.strip() else self.fallback_response
        except Exception as e:
            print(f"Error: {e}")
            return "I encountered an error while processing your request. Please try again."

if __name__ == "__main__":
    chatbot = Chatbot()

    print("Welcome to CHAT! Type your question or 'q' to quit.")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["q", "quit"]:
            print("Goodbye!")
            break

        response = chatbot.get_response(query)
        print("CHAT:", response, "\n")
