import os
from random import sample

import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# Step1: Initialize the google model components(The brain & The measure)
# ==================================================
# The LLM to synthesize the final answers
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# The Embedding model to turn sentences into mathematical vector coordinates
Settings.embed_model = GoogleGenAIEmbedding(
    model="models/text-embedding-004",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# ==================================================
# Step2: Create simple knowledge text file(The Source material)
# ==================================================
os.makedirs("my_data", exist_ok=True)
sample_text = """
Artificial Intelligence (AI) has significantly transformed the modern world by creating new opportunities for development, innovation, and career growth across various industries. With the rapid advancement of AI technologies such as machine learning, deep learning, natural language processing, computer vision, and robotics, organizations are increasingly adopting intelligent systems to improve efficiency, automate repetitive tasks, and enhance decision-making processes. This technological evolution has led to the emergence of numerous career opportunities in fields such as software development, data science, AI engineering, machine learning engineering, robotics, cybersecurity, and cloud computing. Professionals with AI-related skills are highly valued because they can build intelligent applications, analyze large-scale data, develop predictive models, and create automation solutions for real-world problems. Beyond technical roles, AI also creates opportunities in business strategy, product management, research, consulting, healthcare, education, and finance, where professionals leverage AI tools to drive innovation and improve productivity. Developers now have access to powerful AI frameworks and platforms that accelerate software development, enabling them to build smarter applications with personalized user experiences. At the same time, AI encourages continuous learning, as professionals must stay updated with evolving tools, algorithms, and industry trends. As AI continues to grow, individuals who develop strong technical foundations, problem-solving abilities, and adaptability will have a significant advantage in the job market. Overall, AI is not only reshaping the future of technology but also opening endless possibilities for career development, making it one of the most promising and impactful fields of the modern era.

"""
with open("my_data/sample_text.txt", "w", encoding="utf-8") as f:
    f.write(sample_text.strip())

print("[System] Local .txt knowledge file successfully created in './my_data/' folder.")

# ==================================================
# Step3: setup persistent chromadb directory (The storage on disk)
# ==================================================
# This initializes a persistent folder 'chroma_db_storage' right on your hard drive

db_client = chromadb.PersistentClient(path="./chroma_db_storage")

# Create or fetch a table (collection) inside chromadb to hold vector numbers
chroma_collection = db_client.get_or_create_collection("ai_knowledge_base")

# Construct the vector store layer
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==================================================
# Step4: READ, EMBED, INDEX DATA
# ==================================================
# Load your local documents from the folder
documents = SimpleDirectoryReader("./my_data").load_data()

# Process data: read text -> calculate numbers via embed_model -> save to chromadb folder
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("[System] Text embedded and permanently saved to disk in './chroma_db_storage/'.")

# ==================================================
# Step5: Query the rag engine
# ==================================================
query_engine = index.as_query_engine()

# Test case query
test_query = "How has AI transformed the modern world?"
print(f"\n[User Question]: {test_query}")

# The Engine searches chromadb fpr matching numbers, grabs text snippets and feeds them to gemini
response = query_engine.query(test_query)
print(f"[AI Generated Answer]: \n{response}")