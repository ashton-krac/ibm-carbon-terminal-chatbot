# carbon_chatbot.py

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

class CarbonDesignChat:
    def __init__(self, vector_db_path="./vector_db"):
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=OpenAIEmbeddings()
        )
        
        # Set streaming to True
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=.1,
            streaming=True
        )
    
    def ask(self, question):
        # Retrieve relevant documentation
        docs = self.vectorstore.similarity_search(question, k=2)
        
        # Format the retrieved content
        context = "\n".join(
            f"From {doc.metadata['title']}:\n{doc.page_content}\n"
            for doc in docs
        )
        
        # Augment the prompt with retrieved documentation
        prompt = f"""You are an expert on the IBM Carbon Design System. 
        Use the following documentation to answer the question accurately.
        If the specific information isn't in the documentation, say so.
        Give a URL for more information on the related topic, at hand.
        
        Documentation:
        {context}
        
        Question: {question}
        """
        
        # Stream the response
        print("\nAnswer: ", end="", flush=True)
        response = ""
        for chunk in self.llm.stream(prompt):
            chunk_text = chunk.content
            print(chunk_text, end="", flush=True)
            response += chunk_text
        print()  # New line after response
        return response

def main():
    # Check if vector database exists
    if not os.path.exists("./vector_db"):
        print("Error: Vector database not found!")
        print("Please run setup_carbon_db.py first to create the database.")
        return
        
    # Initialize chatbot
    chatbot = CarbonDesignChat()
    
    print("\nCarbon Design System ChatBot (type 'exit' to quit)")
    print("----------------------------------------")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if question:
            chatbot.ask(question)

if __name__ == "__main__":
    main()