# Agentic RAG with CrewAI + Gemini

A 2-agent RAG system over a CSV of support tickets:

1. Retriever Agent - searches a local FAISS vector index for relevant tickets
2. Support Analyst Agent - reasons over retrieved tickets and writes a grounded, cited answer

Both agents run on Gemini (fast, cheap inference) via CrewAI's built-in LiteLLM layer.
Embeddings run locally and free via sentence-transformers (no API cost for retrieval).

## Project structure

- ingest.py - builds the FAISS index from your CSV (run once / on data change)
- tools.py - the CrewAI tool the Retriever agent uses to search the index
- crew.py - agents, tasks, crew definition, and CLI entrypoint
- tickets.csv - sample data, replace with your real CSV
- requirements.txt
- .env.example

## Setup

1. Create a virtual environment (recommended):

   python3 -m venv venv
   source venv/bin/activate    (on Windows: venv\Scripts\activate)

2. Install dependencies:

   pip install -r requirements.txt

3. Set your GEMINI API key:

   cp .env.example .env

   Edit .env and paste your key from https://console.groq.com/keys

   GEMINI_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   CSV_PATH=tickets.csv
   GEMINI_MODEL=gemini-2.5-flash

4. Replace tickets.csv with your real data. Required columns (case-sensitive):
   - Body - the ticket text
   - Department
   - Priority
   - Tags - a stringified list like ['Billing', 'Account']

5. Build the FAISS index (run this once, and again whenever the CSV changes):

   python ingest.py

   This creates a faiss_index/ folder with the vector index and metadata.

## Run

   python crew.py "What kind of issues get marked high priority?"

or run interactively:

   python crew.py

You'll see both agents' reasoning in the terminal (verbose mode), followed by a final structured answer.

## How it works

1. User asks a question.
2. The Retriever Agent calls the search_tickets tool, which embeds the query
   locally and searches the FAISS index built from your CSV.
3. The raw retrieved tickets (department, priority, tags, body) are passed
   as context to the Analyst Agent.
4. The Analyst Agent, running on Gemini Api, reasons over only that retrieved
   evidence and writes the final answer.

This is "agentic" RAG (rather than plain RAG) because:
- The Retriever agent can decide to re-search with a rephrased query if
  results look weak, instead of blindly returning top-k once.
- The Analyst agent is instructed to flag insufficient evidence rather
  than hallucinate an answer, and must cite which retrieved ticket(s)
  support each claim.



