# GeoIntelBot: Intelligent Q&A Bot for MOSDAC Documents

GeoIntelBot is an intelligent bot designed to answer questions about documents from [MOSDAC](https://mosdac.gov.in/). It automatically crawls the MOSDAC website, extracts all PDF documents, processes and chunks their content, and builds a knowledge graph using Gemini LLM and Neo4j. This enables advanced, structured, and context-aware question answering over the MOSDAC document corpus.

## What Can GeoIntelBot Do?
- **Answer questions about MOSDAC policies, reports, manuals, and more**
- **Find facts, relationships, and summaries from official MOSDAC PDFs**
- **Support deep, context-rich queries using a knowledge graph**

## How It Works
1. **Crawling:** Discovers and collects all PDF links from the MOSDAC website.
2. **Extraction:** Downloads and extracts text from each PDF.
3. **Chunking & Preprocessing:** Cleans and splits text for LLM processing.
4. **Knowledge Graph Construction:** Uses Gemini LLM to extract atomic facts and key elements, then ingests them into a Neo4j graph.
5. **Question Answering:** (Planned) Enables users to query the knowledge graph for answers about MOSDAC documents.

## Directory Structure
```
GeoIntelBot/
  main.py                        # Main pipeline: crawl, extract, preprocess
  requirements.txt               # Python dependencies
  src/
    webscraper.py                # Web crawling, PDF extraction, LLM chunking
    prompts.py, schema.py        # LLM prompt and schema definitions
  utils/
    url_utils.py, file_utils.py, log_utils.py
  knowledge_graph_handler/
    handler.py                   # Knowledge graph ingestion (Neo4j + Gemini)
  output/
    mosdac_pdfs.jsonl            # Discovered PDF links
    downloaded_pdfs/             # Downloaded PDF files
    mosdac_pdfs_text.jsonl       # Extracted PDF text
    llm_ready_output.jsonl       # LLM-chunked, cleaned text
  logs/                          # Crawl and ingestion logs
```

## Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd GeoIntelBot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Build the Knowledge Base from MOSDAC
Run the main pipeline to crawl, download, and preprocess MOSDAC PDFs:
```sh
python main.py --start-url "https://mosdac.gov.in/" --max-depth 3 --max-concurrent 10
```
- `--start-url`: Starting URL for crawling (default: MOSDAC homepage)
- `--max-depth`: Maximum crawl depth (default: 3)
- `--max-concurrent`: Maximum concurrent requests (default: 10)

**Outputs:**
- `output/mosdac_pdfs.jsonl`: List of discovered PDF links
- `output/downloaded_pdfs/`: Downloaded PDF files
- `output/mosdac_pdfs_text.jsonl`: Extracted text from PDFs
- `output/llm_ready_output.jsonl`: Cleaned, chunked text for LLM/graph ingestion

### 2. Ingest into the Knowledge Graph
1. Set up a `.env` file with your Neo4j and Gemini credentials:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   GEMINI_API_KEY=your_gemini_api_key
   ```
2. Run the handler to ingest processed data:
   ```sh
   python knowledge_graph_handler/handler.py
   ```
   This will delete the existing graph and ingest all chunks from `output/llm_ready_output.jsonl`.

### 3. (Planned) Ask Questions About MOSDAC
- A conversational interface for querying the knowledge graph is under development.
- The bot will support natural language Q&A over all ingested MOSDAC documents.

## Extending the Project
- Utility functions are in `utils/` for easy reuse and extension.
- Prompts and schemas for LLM extraction are in `src/prompts.py` and `src/schema.py`.
- See `examples/` for Jupyter notebook usage and advanced ingestion.

## Requirements
See `requirements.txt` for all dependencies (crawl4ai, langchain, pymupdf, etc).

## License
MIT 