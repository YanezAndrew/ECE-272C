import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from HW2.project.nodes import (
    codegen_node,
    execute_node,
    evaluate_node,
    retry_codegen_node,
    respond_node,
    visualization_node,
)


def should_retry(state: dict) -> str:
    """Route after evaluation: retry once on FAIL, otherwise respond."""
    if state["evaluation"] == "FAIL" and state.get("retry_count", 0) < 1:
        return "retry"
    return "respond"


def build_agent():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("codegen", codegen_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retry_codegen", retry_codegen_node)
    graph.add_node("respond", respond_node)
    graph.add_node("visualize", visualization_node)

    # Set entry point
    graph.set_entry_point("codegen")

    # Define edges
    graph.add_edge("codegen", "execute")
    graph.add_edge("execute", "evaluate")

    # Conditional: after evaluate, either retry or respond
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "retry_codegen",
            "respond": "respond",
        },
    )

    # After retry, execute again
    graph.add_edge("retry_codegen", "execute")

    # After respond, run visualization (optional step before END)
    graph.add_edge("respond", "visualize")
    graph.add_edge("visualize", END)

    return graph.compile()


agent = build_agent()
