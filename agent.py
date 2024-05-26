import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import load_tools, initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from trafilatura import fetch_url, extract
import os
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("CLI_TOKEN")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    use_auth_token=access_token
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=access_token,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=pipe)

tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llm)
search = DuckDuckGoSearchRun()
tools.append(
    Tool(
        name="Search",
        func=search.run,
        description="Use this tool to search the web for the latest information. Best for finding current events, recent studies, or general information."
    ),
)

class DocumentRetriever:
    def __init__(self, embed_model_name="intfloat/multilingual-e5-small", k=3):
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.k = k
        self.db = None

    def create_db(self, texts):
        if not texts:
            self.db = None
        else:
            self.db = FAISS.from_documents(texts, self.embeddings)

    def get_retriever(self):
        if self.db is None:
            raise ValueError("Database has not been created. Call create_db() first.")
        return self.db.as_retriever(search_kwargs={"k": self.k})

def process_url(url):
    try:
        downloaded = fetch_url(url)
        text = extract(downloaded)
        if text:
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            return text_splitter.split_documents([Document(page_content=text)])
    except Exception as e:
        print(f"Error processing URL: {e}")
    return []

def process_pdf(file_path):
    try:
        pdf_loader = PyPDFLoader(file_path)
        pages = pdf_loader.load()
        text = " ".join(page.page_content for page in pages)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents([Document(page_content=text)])
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return []

def run_agent(query, url, pdf):
    url_docs = process_url(url) if url else []
    pdf_docs = process_pdf(pdf) if pdf else []
    docs = url_docs + pdf_docs

    if docs:
        retriever = DocumentRetriever()
        retriever.create_db(docs)
        rag_retriever = retriever.get_retriever()

        reference_tool = Tool(
            name="Reference Documents",
            func=lambda q: RetrievalQA.from_chain_type(
                llm=llm,
                retriever=rag_retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": "Query: {query}\nAnswer:"},
            ).run(q),
            description="Use this tool to find answers specifically by referencing provided documents such as URLs and PDFs. Ideal for retrieving detailed information from specific sources."
        )
        tools.append(reference_tool)
    
    agent = initialize_agent(
        agent="zero-shot-react-description",
        llm=llm,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    response = agent.run(query)
    return response

def main():
    query = input("Query: ")  
    url = input("URL (leave blank if none): ") or None  
    pdf = input("PDF file path (leave blank if none): ") or None
    response = run_agent(query, url, pdf)
    print(response)

if __name__ == "__main__":
    main()


