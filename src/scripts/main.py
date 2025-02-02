
# main.py
import logging
from scripts.huggingface import HuggingFaceLLM
from models.faiss_knowledge import FAISSKnowledgeBase, initialize_knowledge_base

# Configure logging (if not already configured in faiss_knowledge.py or huggingface.py)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    """Main function to handle user interaction and generate responses."""
    try:
        # Initialize components
        llm = HuggingFaceLLM()
        pdf_path = r"src\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (1).pdf"  # Your PDF path
        knowledge_base = initialize_knowledge_base(pdf_path)

        if knowledge_base is None:
            logger.error("Failed to initialize knowledge base. Exiting.")
            return  # Exit the program if KB initialization fails

        while True:  # Allow for multiple queries
            query = input("Enter your medical query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            context_chunks = knowledge_base.search(query)
            context = " ".join(context_chunks)

            response = llm.generate_response(context, query)

            if response: # Check if response generation was successful
                print("\n💡 Medical Assistant Response:\n")
                print(response)
            else:
                print("\n❌ Could not generate a response.") # Handle the case where the LLM returns None

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()