# 🚀 CrewAI-Powered Medical Assistant with Hugging Face

## 📌 Project Overview

This project is a **Medical Assistant** built using **CrewAI, Hugging Face Inference API, and FAISS**, designed to efficiently search and retrieve medical data from PDF books. It provides a user-friendly **Streamlit UI** for interaction. The project leverages **LLM-powered search**, **PDF embeddings with FAISS**, and **task automation with CrewAI**.

## 📂 Project Structure

📂 src/
│── 📂 config/         # YAML configuration files
│   │── agents.yaml   # Agent definitions
│   │── tasks.yaml    # Task definitions
│── 📂 data/           # Store PDF files here ✅
│── 📂 logs/           # Logging files
│   │── logger.py     # Logger setup
│── 📂 models/         # FAISS Knowledge Base
│   │── faiss_knowledge.py # PDF to vector storage
│── 📂 scripts/        # Core scripts
│   │── crew.py       # CrewAI setup
│   │── huggingface_llm.py   # Hugging Face API integration
│   │── main.py       # Entry point (Runs the assistant - for local testing)
│── 📂 ui/             # Streamlit UI
│   │── app.py        # Streamlit frontend
│── .env               # API keys and environment variables
│── requirements.txt   # Python dependencies
│── README.md           # Project documentation


---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/your-repo.git](https://github.com/your-repo.git)
cd crewai-medical-assistant
2️⃣ Create a Virtual Environment & Install Dependencies
Bash

python3 -m venv venv  # Or python -m venv venv depending on your system
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3️⃣ Set Up Environment Variables
Create a .env file in the root directory and add your API keys:

Ini, TOML

HUGGINGFACE_API_KEY=your_huggingface_api_key
# If using sentence-transformers for embeddings (default):
HF_ACCESS_TOKEN=your_huggingface_access_token # Required for sentence-transformers
Important:  You'll need a Hugging Face API key (HUGGINGFACE_API_KEY) for the LLM interaction. If you are using sentence-transformers for creating embeddings (which is the default in faiss_knowledge.py), you also need a Hugging Face access token (HF_ACCESS_TOKEN).  If you use a different method for embeddings (e.g., from langchain), you might not need the access token.

4️⃣ Prepare Data
Place your medical PDFs inside the src/data/ directory.
5️⃣ Run the Application (Locally)
Bash

streamlit run src/ui/app.py
This starts a Streamlit UI where you can interact with the assistant.

6️⃣ Running with main.py (for local testing and development)
The main.py script is provided for local testing and development. It allows you to interact with the core components of your application (LLM, Knowledge base) directly from the command line.

Bash

python src/scripts/main.py
🚀 How the Project Works
🔹 1. Convert PDFs into a Searchable Knowledge Base (FAISS)
models/faiss_knowledge.py extracts text from PDFs using fitz (PyMuPDF).
It splits the extracted text into smaller chunks using langchain.text_splitter.RecursiveCharacterTextSplitter.
Converts the text chunks into vector embeddings using sentence-transformers (or another embedding model you configure).
Stores embeddings in FAISS for efficient similarity search and retrieval. The FAISS index is saved to disk for persistence.
🔹 2. LLM-Powered Response Generation (Hugging Face)
scripts/huggingface_llm.py integrates with the Hugging Face Inference API for LLM interaction.
It constructs prompts that include both the user's query and the relevant context retrieved from the FAISS knowledge base.
Sends the prompt to the Hugging Face Inference API and receives the LLM-generated response.
🔹 3. Task Automation with CrewAI (Optional)
scripts/crew.py demonstrates how you can use CrewAI agents and tasks for more complex workflows. This is currently not integrated into the main Streamlit app flow but is provided as an example for future expansion.
config/agents.yaml and config/tasks.yaml define the agents and tasks.
🔹 4. User Interaction via Streamlit UI
ui/app.py provides a user-friendly interface built with Streamlit.
Users input queries, and the application:
Searches the FAISS knowledge base for relevant context.
Sends the query and context to the Hugging Face Inference API for response generation.
Displays the LLM-generated response in the Streamlit UI.
Shows the reference context retrieved from FAISS (collapsible).
Handles errors gracefully and provides informative messages to the user.
🛠 Key Components
✅ 1. Configuration (YAML Files)
config/agents.yaml: Defines AI agents (e.g., Researcher, Validator) for CrewAI workflows (optional).
config/tasks.yaml: Defines how tasks are distributed among agents (optional).
✅ 2. Hugging Face LLM API Integration
scripts/huggingface_llm.py: Handles the connection and interaction with the Hugging Face Inference API.
✅ 3. FAISS Vector Database
models/faiss_knowledge.py: Implements the FAISS knowledge base for efficient storage and retrieval of medical knowledge.
✅ 4. CrewAI Agent-Based Architecture (Optional)
scripts/crew.py: Demonstrates the use of CrewAI for managing agents and tasks (for more complex workflows).
✅ 5. Streamlit UI
ui/app.py: The main Streamlit application file that handles user interaction and orchestrates the other components.
✅ 6. Logging
logs/logger.py: Sets up logging to capture important information and errors.
🎯 Future Enhancements
🔹 Add speech-to-text support for voice queries.
🔹 Implement real-time updates for new PDFs.
🔹 Deploy the application to a cloud platform (AWS, GCP, or Hugging Face Spaces).
🔹 Implement more sophisticated agent workflows using CrewAI.
🔹 Add a user authentication system.

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Commit your changes (git commit -m "Added new feature").
Push to the branch (git push origin feature-name).
Open a Pull Request.
📞 Contact & Support
For questions, please feel free to reach out via:

📧 Email: your-email@example.com
🐙 GitHub: Your GitHub Profile

🚀 Enjoy Building Your AI-Powered Medical Assistant!