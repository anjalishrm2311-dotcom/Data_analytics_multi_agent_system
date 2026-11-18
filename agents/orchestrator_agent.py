"""
Orchestrator agent.

Wires the profiling, cleaning, and analysis agents together as nodes in a
LangGraph state graph, and uses a Gemini call to explain (in plain English)
why it decided to clean the data or skip cleaning.

If GEMINI_API_KEY is not set, the pipeline still runs end-to-end using a
simple rule (missing values or duplicates > 0 => clean). The LLM is only
used to generate a human-readable explanation, so the orchestrator degrades
gracefully without an API key.
"""

import os
from typing import TypedDict, List

import pandas as pd
from langgraph.graph import StateGraph, END

from agents.cleaning_agent import cleaning_agent

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

def _model() -> str:
    """First model from GEMINI_MODEL in .env, else a fast default."""
    env = os.environ.get("GEMINI_MODEL", "").strip()
    return env.split(",")[0].strip() if env else "gemini-3.5-flash-lite"


# ============================================================
# STATE
# ============================================================

class PipelineState(TypedDict, total=False):
    raw_df: pd.DataFrame
    profiling_report: dict
    needs_cleaning: bool
    cleaned_df: pd.DataFrame
    cleaning_report: dict
    analysis_report: dict
    decision_log: List[str]


# ============================================================
# LLM HELPER
# ============================================================

def _get_llm_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def _ask_llm(prompt: str) -> str:
    client = _get_llm_client()

    if client is None:
        return (
            "(LLM reasoning not available — set GEMINI_API_KEY "
            "in your .env file to enable natural-language explanations.)"
        )

    try:
        response = client.models.generate_content(
            model=_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return (response.text or "").strip() or "(LLM returned an empty response.)"
    except Exception as e:
        return f"(LLM call failed: {e})"


# ============================================================
# NODES
# ============================================================

def profiling_node(state: PipelineState) -> PipelineState:

    df = state["raw_df"]

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.astype(str).duplicated().sum()),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "text_columns": list(df.select_dtypes(include="object").columns),
    }

    state["profiling_report"] = report

    log = state.get("decision_log", [])
    log.append(
        f"🔍 Profiling agent: {report['rows']} rows, {report['columns']} columns, "
        f"{report['missing_values']} missing values, {report['duplicate_rows']} duplicate rows."
    )
    state["decision_log"] = log

    return state


def decide_cleaning_node(state: PipelineState) -> PipelineState:

    report = state["profiling_report"]
    needs_cleaning = report["missing_values"] > 0 or report["duplicate_rows"] > 0
    state["needs_cleaning"] = needs_cleaning

    explanation = _ask_llm(
        "A data profiling agent scanned a dataset and found "
        f"{report['missing_values']} missing values and {report['duplicate_rows']} "
        f"duplicate rows out of {report['rows']} total rows. "
        "In exactly one short sentence, explain to a non-technical user whether "
        "the dataset needs cleaning and why."
    )

    log = state.get("decision_log", [])
    log.append(
        f"🧠 Orchestrator decision: {'cleaning needed' if needs_cleaning else 'no cleaning needed'}. "
        f"{explanation}"
    )
    state["decision_log"] = log

    return state


def route_after_decision(state: PipelineState) -> str:
    return "clean" if state.get("needs_cleaning") else "skip_clean"


def cleaning_node(state: PipelineState) -> PipelineState:

    df = state["raw_df"]
    cleaned_df, report = cleaning_agent(df)

    state["cleaned_df"] = cleaned_df
    state["cleaning_report"] = report

    filled_cols = len([k for k in report if k.endswith("_missing_filled")])

    log = state.get("decision_log", [])
    log.append(
        f"🧹 Cleaning agent: removed {report.get('duplicates_removed', 0)} duplicate rows, "
        f"filled missing values in {filled_cols} column(s)."
    )
    state["decision_log"] = log

    return state


def skip_clean_node(state: PipelineState) -> PipelineState:

    state["cleaned_df"] = state["raw_df"]
    state["cleaning_report"] = {"skipped": True}

    log = state.get("decision_log", [])
    log.append("🧹 Cleaning agent: skipped, dataset was already clean.")
    state["decision_log"] = log

    return state


def analysis_node(state: PipelineState) -> PipelineState:

    df = state["cleaned_df"]
    numeric_cols = df.select_dtypes(include="number").columns

    report = {
        "describe": df.describe(include="all"),
        "correlation": df[numeric_cols].corr() if len(numeric_cols) >= 2 else None,
    }

    state["analysis_report"] = report

    log = state.get("decision_log", [])
    log.append("📊 Analysis agent: computed descriptive statistics and correlations.")
    state["decision_log"] = log

    return state


# ============================================================
# GRAPH
# ============================================================

def build_pipeline():

    graph = StateGraph(PipelineState)

    graph.add_node("profiling", profiling_node)
    graph.add_node("decide_cleaning", decide_cleaning_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("skip_clean", skip_clean_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("profiling")
    graph.add_edge("profiling", "decide_cleaning")

    graph.add_conditional_edges(
        "decide_cleaning",
        route_after_decision,
        {"clean": "cleaning", "skip_clean": "skip_clean"},
    )

    graph.add_edge("cleaning", "analysis")
    graph.add_edge("skip_clean", "analysis")
    graph.add_edge("analysis", END)

    return graph.compile()


def run_pipeline(df: pd.DataFrame) -> PipelineState:
    """
    Runs the full profiling -> decide -> cleaning/skip -> analysis pipeline
    on the given dataframe and returns the final state, including a
    step-by-step decision_log you can show in the UI.
    """
    pipeline = build_pipeline()
    initial_state: PipelineState = {"raw_df": df, "decision_log": []}
    final_state = pipeline.invoke(initial_state)
    return final_state