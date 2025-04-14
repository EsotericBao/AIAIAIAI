from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_chroma import Chroma
from config import MAIN_COLLECTION,COLLECTION, FALLBACK_COLLECTION, Models, MAINDB, CHROMADB, FALLBACKDB
import sys, time, re
NAME = "Cody"

class Chatbot:
    def __init__(self):
        """
        Initialize the chatbot with models, vector store, and prompt template.
        """
        self.models = Models()
        self.embeddings = self.models.embeddings_main
        self.llm = self.models.model_main

        # Initialize vector store
        self.maindb = Chroma(
            collection_name=MAIN_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=MAINDB,
        )
        self.chromadb = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMADB,
        )

        self.fallbackdb = Chroma(
            collection_name=FALLBACK_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=FALLBACKDB,
        )


        # Define prompt template
        self.prompt = ChatPromptTemplate.from_messages(
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
                    - respond in less than 3 sentences

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
        main_retriever = self.maindb.as_retriever(kwargs={"k": 20})
        chroma_retriever = self.chromadb.as_retriever(kwargs={"k": 3})

        # Define weights (higher means more priority)
        weights = [0.9, 0.1]  # maindb is prioritise
        ensemble_retriever = EnsembleRetriever(
        retrievers=[main_retriever, chroma_retriever],
        weights=weights
        )
        # Define the retrieval chain
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(ensemble_retriever, self.combine_docs_chain)

        # Define fallback chatbot prompt
        self.fallback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """Your name is CHAT, a helpful assistant.
                    - If a user asks a question, respond with accurate and clear information.
                    - Keep your responses short, direct, and relevant to the context.
                    - Avoid repeating information unnecessarily or introducing yourself repeatedly.
                    - Dont discuss politics
                """),
                ("human", 
                 """Use the following question and context to generate a response:
                 Question: {input}
                 Context: {context}
                 """)
            ]
        )
        # Define fallback chatbot chain
        self.fallback_retriever = self.fallbackdb.as_retriever(kwargs={"k": 20})
        self.fallback_docs_chain = create_stuff_documents_chain(self.models.model_fallback, self.fallback_prompt)
        self.fallback_retrieval_chain = create_retrieval_chain(self.fallback_retriever, self.fallback_docs_chain)

        
        print("\n")

    def elapsed_time(self):
        print(time.time() - self.time, end='\n')

    def remove_think(self,response):
        l = response.split("</think>")
        clean = l[len(l)-1]
        return clean.strip("\n")
    
    def get_response(self, query):
        """

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        print("Getting response...")
        self.time = time.time()
        try:
            # Attempt response from SPF retrieval-based chatbot
            result = self.retrieval_chain.invoke({"input": query})
            self.elapsed_time()
            response = result.get("answer", "").strip()
            print(response, end='\n\n')
            fallback_flag = False
            fallback_key = [
                "not mentioned",
                "I couldn't find relevant"
            ]
            for i in fallback_key:
                if i in response:
                    fallback_flag = True
            if not response or fallback_flag:
                self.elapsed_time()
                print("⚠️ No relevant data found. Using fallback chatbot...")
                fallback_response = self.fallback_retrieval_chain.invoke({"input": query}).get("answer", "").strip()
                return fallback_response if fallback_response else "I couldn't retrieve an answer."
            
            return self.remove_think(response)

        except Exception as e:
            print(f"Error in response generation: {e}")
            return "I'm sorry, an error occurred while processing your request."
        
    
            

if __name__ == "__main__":
    chatbot = Chatbot()
    print(f"Welcome to {NAME}! Ask your question or type 'q', 'quit', or 'exit' to end the session.")
    print("Type 'clear' or 'reset' to start a new session.")


    while True:
        query = input("User: ").strip()
        
        if query.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        response = chatbot.get_response(query)
        answer = chatbot.remove_think(response)
        # Get response from chatbot
        print(f"{NAME}: {answer}\n")
        #chatbot.stream_response(response)
        
