from pathlib import Path
from bs4 import BeautifulSoup
from embed import EmbedChunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ray
from functools import partial
import logging
import uuid
import chromadb

class HTMLProcessor:
    def __init__(self, root_dir, model_name, db_path="./chromdb"):
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(root_dir)
        self.ds = ray.data.from_items([{"path": path} for path in self.root_dir.rglob("*.html") if not path.is_dir()])
        self.model_name = model_name
        self.db_path = db_path

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        # You can configure the logging format and handlers as needed

    def extract_sections(self, record):
        with open(record["path"], "r", encoding="latin-1") as html_file:
            html_content = html_file.read()
            soup = BeautifulSoup(html_content, "html.parser")
        
        sections = soup.find_all("p")
        section_text = ""
        seen_texts = set()

        for section in sections:
            text = section.get_text(strip=True)
            if text and text not in seen_texts:
                seen_texts.add(text)
                section_text += text + " "

        section_text = section_text.strip()

        if section_text:
            uri = self.path_to_uri(path=record["path"])
            return [{"source": f"{uri}", "text": section_text}]
        else:
            return []

    def path_to_uri(self, path, scheme="https://", domain="engineering.nyu.edu"):
        uri = scheme + domain + str(path).split(domain)[-1]
        return uri[:-5] if uri.endswith(".html") else uri

    def chunk_section(self, section, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len)
        chunks = text_splitter.create_documents(
            texts=[section["text"]], 
            metadatas=[{"source": section["source"]}])
        return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]

    def process_html(self, chunk_size=500, chunk_overlap=50, batch_size=128):
        self.setup_logging()

        # Extract sections
        sections_ds = self.ds.flat_map(self.extract_sections)
        self.logger.info(f"Number of sections: {sections_ds.count()}")

        # Scale chunking
        chunks_ds = sections_ds.flat_map(partial(
            self.chunk_section, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap))
        self.logger.info(f"Number of chunks: {chunks_ds.count()}")

        # Create an instance of EmbedChunks
        embed_chunks_instance = EmbedChunks(model_name=self.model_name)

        def chunk_section(batch, hf_embed_model):
            embeddings = hf_embed_model.embed_documents(batch["text"])
            return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}

        embedded_chunks = chunks_ds.map_batches(
            chunk_section,
            fn_kwargs={"hf_embed_model": embed_chunks_instance.embedding_model},
            batch_size=batch_size)
        self.logger.info(f"Number of embedded chunks: {embedded_chunks.count()}")

        embed_df = embedded_chunks.to_pandas()

        # Generate a unique string ID for each row
        embed_df['UniqueID'] = embed_df.apply(lambda row: str(uuid.uuid4()), axis=1)

        # Convert the links into a dictionary
        embed_df['source'] = embed_df['source'].apply(lambda link: {'source': link})

        # Initialize and store results in ChromaDB
        chroma_db = ChromaDB(db_path=self.db_path)
        chroma_db.store_results(embed_df)

class ChromaDB:
    def __init__(self, db_path):
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="test")

    def store_results(self, embed_df):
        self.logger.info("Storing results in ChromaDB")
        self.collection.add(
            ids=embed_df.UniqueID.tolist(),
            documents=embed_df.text.tolist(),
            embeddings=embed_df.embeddings.tolist(),
            metadatas=embed_df.source.tolist())

if __name__ == "__main__":
    html_processor = HTMLProcessor(root_dir='/Users/himanshukumar/Desktop/nyuinfo_rag', model_name="sentence-transformers/all-MiniLM-l6-v2")
    html_processor.process_html()
