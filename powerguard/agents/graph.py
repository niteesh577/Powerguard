"""
agents/graph.py
===============
LangGraph StateGraph definition.

Pipeline:
  START → biomech_agent → fatigue_agent → risk_agent → END

All nodes receive and return the AgentState TypedDict.
The graph is compiled once at module import time and reused across requests.
"""

from langgraph.graph import StateGraph, START, END

from schemas.agent_state import AgentState
from agents.biomech_agent import biomech_agent_node
from agents.fatigue_agent import fatigue_agent_node
from agents.risk_agent import risk_agent_node
from observability.logger import get_logger

logger = get_logger("agent_graph")


def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("biomech_agent",  biomech_agent_node)
    builder.add_node("fatigue_agent",  fatigue_agent_node)
    builder.add_node("risk_agent",     risk_agent_node)

    builder.add_edge(START,           "biomech_agent")
    builder.add_edge("biomech_agent", "fatigue_agent")
    builder.add_edge("fatigue_agent", "risk_agent")
    builder.add_edge("risk_agent",    END)

    return builder.compile()


# Compiled graph — singleton, reused across all requests
_graph = _build_graph()
logger.info("agent_graph.compiled", nodes=["biomech_agent", "fatigue_agent", "risk_agent"])


def get_graph():
    """Return the compiled LangGraph agent pipeline."""
    return _graph


async def run_pipeline(initial_state: AgentState) -> AgentState:
    """
    Run the full agent pipeline asynchronously.
    Returns the final state after all 3 agents have run.
    """
    logger.info(
        "pipeline.start",
        trace_id=initial_state.get("trace_id"),
        lift=initial_state.get("lift_type"),
        user=initial_state.get("user_id"),
    )
    final_state = await _graph.ainvoke(initial_state)
    logger.info(
        "pipeline.complete",
        trace_id=initial_state.get("trace_id"),
        risk_level=final_state.get("risk_output", {}).get("risk_level", "?"),
    )
    return final_state
