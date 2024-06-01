from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # load files from directories
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS  

DATA_PATH = "data/"

DB_FAISS_PATH = "vectorstores/db_faiss"

# create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,glob='*.pdf',loader_cls=PyPDFLoader) 
    #glob - to match file type in the directory
    #loader_cls = specifies loader class for loading data from pdf file
    
    documents = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap = 50)
    #chunk_size - dividing text in smaller chunks
    #chunk_overlap - overlap between chunks ,each chunk overlap other consecutive chunk by specified characters
    
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device':'cuda'})
    #model_name - name of the embedding model
    
    db = FAISS.from_documents(texts, embeddings)
    #create vector and store it in vector store
    
    db.save_local(DB_FAISS_PATH)
    #save_local - saves the vector database to disk
    
if __name__ == '__main__':
    create_vector_db()