
    
   
# app.py
import sys
import os
import streamlit as st
from typing import List
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from logs.logger import log_info, log_error, log_debug, log_exception  # Make sure you have a logger module
from scripts.huggingface import HuggingFaceLLM
from models.faiss_knowledge import FAISSKnowledgeBase, initialize_knowledge_base  # Use initialize_knowledge_base


def initialize_components():
    """Initialize LLM and knowledge base components using session state."""
    if 'llm' not in st.session_state:
        log_info("Initializing HuggingFace LLM")
        st.session_state.llm = HuggingFaceLLM()

    if 'knowledge_base' not in st.session_state:
        log_info("Initializing Knowledge Base")
        pdf_path = r"src\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (1).pdf"  # Define PDF path here
        st.session_state.knowledge_base = initialize_knowledge_base(pdf_path)

        if st.session_state.knowledge_base is None:  # Handle KB initialization failure
            st.error("Knowledge base initialization failed. Please check logs.")
            st.stop()  # Stop execution if KB fails

    return st.session_state.llm, st.session_state.knowledge_base


def generate_prompt(context: List[str], query: str) -> str:
    """Generate a structured prompt for the LLM."""
    context_text = " ".join(context) if isinstance(context, list) else context
    return f"""Based on the following medical context, please answer the question. 
    If you cannot find relevant information in the context, please indicate that clearly.

    Context: {context_text}

    Question: {query}

    Answer:"""


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Medical Assistant",
        page_icon="🩺",
        layout="wide"
    )

    st.title("🩺 Medical Assistant")
    st.markdown("---")

    try:
        llm, knowledge_base = initialize_components()

        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown("""
            This Medical Assistant uses advanced AI to help answer your medical queries.
            It searches through medical documentation to provide accurate information.
            """)
            st.markdown("---")
            st.subheader("🤖 Model Information")
            st.write(f"Current Model: {llm.model}")

        query = st.text_input("Enter your medical query:", key="query_input")

        if query:
            log_info(f"Processing query: {query}")

            with st.spinner("🔍 Searching medical knowledge base..."):
                context_chunks = knowledge_base.search(query)
                log_debug(f"Retrieved context chunks: {len(context_chunks)} pieces")

            with st.spinner("💭 Generating response..."):
                prompt = generate_prompt(context_chunks, query)
                print(f"Prompt being sent to LLM: {prompt}")  # Debug print
                print(f"Context being sent to LLM: {context_chunks}") # Debug print
                print(f"Query being sent to LLM: {query}") # Debug print
                response = llm.generate_response(context_chunks, query)  # Correct call!
                print(f"Raw Response from LLM: {response}") # Debug print
                log_debug(f"Generated response: {response}")

            st.markdown("### 💡 Answer:")

            if response:
                if isinstance(response, list) and response:
                    answer = response[0].get('generated_text', '').strip()
                elif isinstance(response, dict):
                    answer = response.get('generated_text', '').strip()
                elif isinstance(response, str):
                    answer = response.strip()
                else:
                    answer = "Unexpected response format from LLM."
                st.markdown(answer)
            else:
                st.markdown("Could not generate a response.")

            with st.expander("📚 Reference Context"):
                st.markdown("The response was generated based on the following context:")
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**Chunk {i}:**\n{chunk}\n---")

    except Exception as e:
        log_exception("Application error", e)  # Corrected: Pass the exception object 'e'
        st.error(f"An error occurred: {type(e).__name__}: {str(e)}")  # This will now work correctly
        st.info("Please check the logs for more details or try a different query.")


if __name__ == "__main__":
    main()