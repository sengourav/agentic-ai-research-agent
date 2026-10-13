"""LangGraph-based Agentic AI Research Agent.

Required environment variables: OPENAI_API_KEY and TAVILY_API_KEY.
Optional: OPENAI_MODEL (defaults to gpt-4o-mini).
"""
from __future__ import annotations

import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient


class AgentState(TypedDict, total=False):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[dict[str, Any]]
    revision_number: int
    max_revisions: int
    needs_more_research: bool
    missing_evidence: list[str]


class Queries(BaseModel):
    queries: list[str] = Field(description="Up to three distinct web search queries")


class CritiqueResult(BaseModel):
    critique: str = Field(description="Concise assessment of the research brief")
    needs_more_research: bool = Field(description="Whether material evidence gaps remain")
    missing_evidence: list[str] = Field(default_factory=list, description="Specific gaps to investigate")


PLAN_PROMPT = """You are a research planning agent. Break the user's question into a concise plan with main sub-questions and evidence needed. Do not write the final report yet."""
RESEARCH_PLAN_PROMPT = """Generate up to 3 focused and distinct web search queries covering different parts of the research plan. Prefer credible sources and recent information when relevant. Return only the structured query list."""
WRITER_PROMPT = """Write a clear, balanced research brief answering the question and following the plan. Base factual claims on the supplied web evidence; do not invent facts or URLs. Use inline citations [1], [2] that match the numbered evidence sources and end with a Sources section listing their titles and URLs. Note uncertainty, conflicting evidence, limitations, and unanswered questions.

WEB EVIDENCE:
------
{content}
"""
REFLECTION_PROMPT = """Critically evaluate whether the brief answers the question, covers the plan, supports important claims with citations, and identifies evidence gaps or conflicts. Return a concise critique, needs_more_research, and specific missing_evidence. Request more research only when web searches could materially address a gap."""
RESEARCH_CRITIQUE_PROMPT = """Generate up to 3 targeted web search queries based on the critic's feedback and missing evidence. Prioritize missing sub-questions, disputed claims, and authoritative sources. Return only the structured query list."""


def structured_value(result: Any, key: str, default: Any = None) -> Any:
    """Read a structured-output field from either a dict or Pydantic model."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def canonical_url(source: dict[str, Any]) -> str:
    raw_url = str(source.get("url", "")).strip()
    if not raw_url:
        return ""
    parts = urlsplit(raw_url)
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), parts.query, ""))


def merge_unique_sources(existing: list[dict[str, Any]], new_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge source records while deduplicating by normalized URL."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (existing or []) + (new_sources or []):
        if not isinstance(source, dict) or not source.get("url"):
            continue
        key = canonical_url(source) or str(source.get("url"))
        if key not in seen:
            seen.add(key)
            merged.append(source)
    return merged


def run_search(tavily: TavilyClient, query: str, max_results: int = 3) -> list[dict[str, Any]]:
    response = tavily.search(query=query, max_results=max_results)
    sources = []
    for item in response.get("results", []):
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        sources.append({
            "query": query,
            "title": item.get("title") or "Untitled",
            "url": url,
            "published_date": item.get("published_date"),
            "content": item.get("content") or "",
            "score": item.get("score"),
        })
    return sources


def search_queries(tavily: TavilyClient, queries: list[str]) -> list[dict[str, Any]]:
    clean_queries = list(dict.fromkeys(q.strip() for q in (queries or []) if q and q.strip()))[:3]
    if not clean_queries:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(3, len(clean_queries))) as executor:
        futures = {executor.submit(run_search, tavily, query): query for query in clean_queries}
        for future, query in [(future, futures[future]) for future in futures]:
            try:
                grouped[query] = future.result()
            except Exception as exc:
                print(f"Search failed for {query!r}: {type(exc).__name__}: {exc}")
                grouped[query] = []
    return [source for query in clean_queries for source in grouped.get(query, [])]


def format_sources_for_prompt(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "No usable web evidence was retrieved. Do not present unsupported claims as verified facts."
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            f"[{index}]\nTitle: {source.get('title', 'Untitled')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Published date: {source.get('published_date') or 'not provided'}\n"
            f"Evidence snippet: {source.get('content', '')}"
        )
    return "\n\n".join(blocks)


def check_research_brief(draft: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    citation_ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", draft or "")})
    source_section = bool(re.search(r"(?im)^\s*(sources|references)\s*:?[ \t]*$", draft or ""))
    return {
        "has_inline_citations": bool(citation_ids),
        "citation_ids_used": citation_ids,
        "available_source_count": len(sources or []),
        "has_sources_section": source_section,
        "citation_ids_out_of_range": [n for n in citation_ids if n > len(sources or [])],
        "note": "These checks inspect formatting and numbering only; they do not verify factual support.",
    }


def export_research_brief(draft: str, filename: str = "research_brief.md") -> str:
    path = Path(filename)
    path.write_text(draft or "", encoding="utf-8")
    return str(path)


def build_graph():
    """Initialize clients and compile the graph. Keys are loaded from .env or shell."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is missing. Copy .env.example to .env and add your key.")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    tavily = TavilyClient(api_key=tavily_key)
    connection = sqlite3.connect(os.getenv("CHECKPOINT_DB", "research_agent_checkpoints.sqlite"), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    def plan_node(state: AgentState):
        response = model.invoke([SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])])
        return {"plan": response.content}

    def research_plan_node(state: AgentState):
        result = model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=f"Question: {state['task']}\nPlan: {state.get('plan', '')}"),
        ])
        queries = structured_value(result, "queries", []) or []
        new_sources = search_queries(tavily, queries)
        print(f"Initial research retrieved {len(new_sources)} source records.")
        return {"content": merge_unique_sources(state.get("content", []), new_sources)}

    def generation_node(state: AgentState):
        evidence = format_sources_for_prompt(state.get("content", []))
        response = model.invoke([
            SystemMessage(content=WRITER_PROMPT.format(content=evidence)),
            HumanMessage(content=(
                f"Research question: {state['task']}\nPlan: {state.get('plan', '')}\n"
                f"Previous critique: {state.get('critique', '')}\n"
                f"Evidence gaps to address: {state.get('missing_evidence', [])}"
            )),
        ])
        return {"draft": response.content, "revision_number": state.get("revision_number", 0) + 1}

    def reflection_node(state: AgentState):
        result = model.with_structured_output(CritiqueResult).invoke([
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=(
                f"Question: {state['task']}\nPlan: {state.get('plan', '')}\n"
                f"Brief: {state.get('draft', '')}\n"
                f"Evidence: {format_sources_for_prompt(state.get('content', []))}"
            )),
        ])
        return {
            "critique": structured_value(result, "critique", ""),
            "needs_more_research": bool(structured_value(result, "needs_more_research", False)),
            "missing_evidence": structured_value(result, "missing_evidence", []) or [],
        }

    def research_critique_node(state: AgentState):
        result = model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=(
                f"Question: {state['task']}\nPlan: {state.get('plan', '')}\n"
                f"Critique: {state.get('critique', '')}\n"
                f"Missing evidence: {state.get('missing_evidence', [])}"
            )),
        ])
        queries = structured_value(result, "queries", []) or []
        new_sources = search_queries(tavily, queries)
        print(f"Follow-up research retrieved {len(new_sources)} source records.")
        return {"content": merge_unique_sources(state.get("content", []), new_sources)}

    def route_after_critique(state: AgentState):
        if state.get("revision_number", 0) >= state.get("max_revisions", 2):
            return END
        if state.get("needs_more_research", False):
            return "research_critique"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_node)
    builder.add_node("research_plan", research_plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_critique", research_critique_node)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", route_after_critique, {END: END, "research_critique": "research_critique"})
    builder.add_edge("research_critique", "generate")
    return builder.compile(checkpointer=checkpointer), connection


def run_task(task: str, thread_id: str = "research-demo-1", max_revisions: int = 2) -> dict[str, Any]:
    graph, connection = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "task": task,
        "content": [],
        "revision_number": 0,
        "max_revisions": max_revisions,
        "needs_more_research": False,
        "missing_evidence": [],
    }
    try:
        for update in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in update.items():
                print(f"\n--- {node_name} ---")
                if "plan" in node_update:
                    print(node_update["plan"])
                elif "critique" in node_update:
                    print(node_update["critique"])
                    print("Needs more research:", node_update.get("needs_more_research"))
                elif "draft" in node_update:
                    draft = node_update["draft"]
                    print(draft[:1500] + ("..." if len(draft) > 1500 else ""))
        final_state = graph.get_state(config).values
        draft = final_state.get("draft", "")
        print("\n===== FINAL RESEARCH BRIEF =====\n")
        print(draft or "No research brief was produced.")
        print("\n===== BASIC QUALITY CHECKS =====")
        print(check_research_brief(draft, final_state.get("content", [])))
        return final_state
    finally:
        connection.close()


if __name__ == "__main__":
    default_task = (
        "Research recent advances in agentic AI, including key approaches, "
        "real-world applications, limitations, and open challenges."
    )
    task = os.getenv("RESEARCH_TASK", default_task)
    thread_id = os.getenv("RESEARCH_THREAD_ID", "research-demo-1")
    final_state = run_task(task, thread_id=thread_id)
    if os.getenv("EXPORT_MARKDOWN", "true").lower() == "true":
        output = export_research_brief(final_state.get("draft", ""))
        print(f"\nSaved Markdown report to: {output}")
