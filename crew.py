"""
crew.py

Defines the agentic RAG crew:
  1. Retriever Agent  - searches the FAISS knowledge base for relevant tickets
  2. Support Analyst  - reasons over retrieved tickets and writes a grounded answer

Both agents run on Gemini via LiteLLM (CrewAI's default LLM layer).
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import search_tickets

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Copy .env.example to .env and add your key "
        "from https://aistudio.google.com/apikey"
    )

# CrewAI uses LiteLLM under the hood. The "gemini/" prefix tells LiteLLM
# which provider to route to. LiteLLM reads GEMINI_API_KEY from the
# environment automatically.
llm = LLM(
    model=f"gemini/{GEMINI_MODEL}",
    temperature=0.2,
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

retriever_agent = Agent(
    role="Ticket Retrieval Specialist",
    goal=(
        "Find the most relevant past support tickets for a given query by "
        "searching the knowledge base. Always search before concluding "
        "nothing relevant exists, and try rephrased queries if the first "
        "search returns weak results."
    ),
    backstory=(
        "You are a meticulous research assistant inside a customer support "
        "team. You know the ticket knowledge base inside out and your only "
        "job is to surface the most relevant historical tickets — including "
        "their department, priority, and tags — so the analyst can use them "
        "as grounded evidence."
    ),
    tools=[search_tickets],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

analyst_agent = Agent(
    role="Senior Support Analyst",
    goal=(
        "Answer the user's question accurately using ONLY the ticket "
        "evidence provided by the Retrieval Specialist. Clearly state "
        "department/priority patterns when relevant, and say so explicitly "
        "if the evidence is insufficient rather than guessing."
    ),
    backstory=(
        "You are an experienced support operations analyst. You're known "
        "for never making claims you can't back up with the retrieved "
        "tickets, and for writing answers that are concise, structured, "
        "and genuinely useful to whoever asked the question."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

retrieval_task = Task(
    description=(
        "Search the ticket knowledge base for tickets relevant to this "
        "question: '{query}'. Use the Ticket Knowledge Base Search tool. "
        "If initial results seem weak or off-topic, try at least one "
        "rephrased search before giving up. Return the raw retrieved "
        "tickets (with department, priority, tags, body) — do not "
        "summarize or interpret them yet."
    ),
    expected_output=(
        "A list of the most relevant tickets retrieved from the knowledge "
        "base, each with department, priority, tags, and body text intact."
    ),
    agent=retriever_agent,
)

analysis_task = Task(
    description=(
        "Using ONLY the tickets retrieved in the previous step, answer "
        "this question thoroughly: '{query}'. "
        "Structure your answer with:\n"
        "1. A direct answer to the question\n"
        "2. Supporting evidence (cite which retrieved ticket(s) you're "
        "drawing from, by department/priority/tags)\n"
        "3. A note if the retrieved evidence is incomplete or doesn't "
        "fully answer the question\n"
        "Do not invent information that isn't in the retrieved tickets."
    ),
    expected_output=(
        "A clear, well-structured answer grounded in the retrieved "
        "tickets, with explicit references to the evidence used."
    ),
    agent=analyst_agent,
    context=[retrieval_task],
)


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

rag_crew = Crew(
    agents=[retriever_agent, analyst_agent],
    tasks=[retrieval_task, analysis_task],
    process=Process.sequential,
    verbose=True,
)


def run_query(query: str) -> str:
    """Convenience wrapper: run the crew on a single query and return the result."""
    result = rag_crew.kickoff(inputs={"query": query})
    return str(result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Ask a question about the support tickets: ")

    print("\n" + "=" * 70)
    print("RUNNING AGENTIC RAG CREW")
    print("=" * 70 + "\n")

    answer = run_query(user_query)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)