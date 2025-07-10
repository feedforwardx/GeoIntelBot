import os
import json
import asyncio

from datetime import datetime
from pydantic import SecretStr
from dotenv import load_dotenv
from utils.file_utils import encode_md5

from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schema import AtomicFact, Extraction
from src.prompts import construction_system, construction_human, import_query, populate_graph_query


# Load environment variables from .env
load_dotenv()

class KnowledgeGraphHandler:
    """
    Handler for Knowledge Graph ingestion and management.
    Usage:
        handler = KnowledgeGraphHandler()
        await handler.ingest_document(text, "docname")
        await handler.ingest_jsonl("output/llm_ready_output.jsonl")
        handler.delete_graph()
    """
    def __init__(self, 
                 model : str = "gemini-1.5-pro", 
                 chunk_size : int = 2000, 
                 chunk_overlap : int = 200):
        """Initialize the Knowledge Graph Handler with necessary configurations."""
        
        # Set default parameters
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load credentials from .env
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password, self.gemini_api_key]):
            raise EnvironmentError("Missing credentials in .env file.")
        
        # Initialize Neo4j Graph
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            refresh_schema=False
        )
        
        # Create constraints
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        
        # LLM setup
        if not self.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=SecretStr(self.gemini_api_key)
        )
        self.construction_prompt = ChatPromptTemplate.from_messages([
            ("system", construction_system),
            ("human", construction_human),
        ])
                
        self.chain = self.construction_prompt | self.llm.with_structured_output(Extraction)

    async def ingest_document(self, 
                              text : str, 
                              document_name : str, 
                              chunk_size : int = None, 
                              chunk_overlap : int = None):
        
        """ Ingest a document into the Knowledge Graph. """
        
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        start = datetime.now()
        print(f"Started extraction at: {start}")
        
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_text(text)
        print(f"Total text chunks: {len(texts)}")
        
        tasks = [
            asyncio.create_task(self.chain.ainvoke({"input": chunk_text}))
            for chunk_text in texts
        ]
        results = await asyncio.gather(*tasks)
        print(f"Finished LLM extraction after: {datetime.now() - start}")
        
        docs = []
        for el in results:
            if isinstance(el, dict):
                docs.append(el)
            else:
                docs.append(el.model_dump())
                
        for index, doc in enumerate(docs):
            doc['chunk_id'] = encode_md5(texts[index])
            doc['chunk_text'] = texts[index]
            doc['index'] = index
            for af in doc["atomic_facts"]:
                af["id"] = encode_md5(af["atomic_fact"])
                
        self.graph.query(import_query,
                        params={"data": docs, "document_name": document_name})
        self.graph.query(populate_graph_query,
                         params={"document_name": document_name})
        print(f"Finished import at: {datetime.now() - start}")

    async def ingest_jsonl(self, 
                           file_path: str, 
                           chunk_size: int = None, 
                           chunk_overlap: int = None):
        
        """ Ingest a JSONL file into the Knowledge Graph. """

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            try:
                record = json.loads(line)
                await self.ingest_document(
                    text=record["text"],
                    document_name=record["id"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                print(f"Processed: {record.get('id')}")
            except Exception as e:
                print(f"Failed to process line: {line[:80]}... Error: {e}")

    def delete_graph(self):
        
        """ Delete the entire Knowledge Graph. """
        
        self.graph.query("MATCH (n) DETACH DELETE n")
        print("Graph deleted successfully.") 

async def main():
        kg = KnowledgeGraphHandler()
        kg.delete_graph()
        await kg.ingest_jsonl("output/modsac_scraped_final.jsonl")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# if __name__ == "__main__":
#     kg = KnowledgeGraphHandler()
#     kg.delete_graph()a
#     kg.ingest_jsonl("output/modsac_scraped_final.jsonl")
