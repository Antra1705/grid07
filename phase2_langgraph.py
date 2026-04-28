from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph


def mock_searxng_search(query: str) -> List[str]:
    """
    Phase 2: mock "web search" tool. Returns hardcoded headlines based on keywords.

    This keeps the graph deterministic and demo-friendly while still showing how a
    search enrichment step would feed extra context into generation.
    """
    q = query.lower()
    if any(k in q for k in ["crypto", "bitcoin", "eth", "solana", "defi"]):
        return [
            "Bitcoin volatility spikes as ETF inflows surge",
            "Ethereum roadmap debate heats up over scaling priorities",
            "DeFi volumes rebound amid renewed risk-on sentiment",
        ]
    if any(k in q for k in ["ai", "llm", "model", "gpu", "inference"]):
        return [
            "New open LLMs narrow the gap on reasoning benchmarks",
            "GPU supply constraints ease, shifting focus to inference efficiency",
            "Enterprises accelerate AI adoption with tighter governance controls",
        ]
    if any(k in q for k in ["markets", "stocks", "rates", "fed", "inflation"]):
        return [
            "Markets price in fewer rate cuts after sticky inflation print",
            "Mega-cap earnings drive index swings in choppy session",
            "Bond yields climb as macro data surprises to upside",
        ]
    return [
        "Headlines are mixed as narratives compete for attention",
        "Analysts debate whether this trend has real staying power",
        "Investors watch for catalysts as volatility stays elevated",
    ]


class BotDraft(TypedDict):
    # Final JSON shape expected from the DraftPost node.
    bot_id: str
    topic: str
    post_content: str


class GraphState(TypedDict, total=False):
    # Shared state passed from node to node through the LangGraph pipeline.
    bot_id: str
    persona: str
    seed_post: str
    topic: str
    do_search: bool
    search_query: str
    search_results: List[str]
    draft_json: Dict[str, Any]


def _guess_topic(seed_post: str) -> str:
    """Map the seed post into a coarse topic that drives search behavior."""
    s = seed_post.lower()
    if any(k in s for k in ["crypto", "bitcoin", "eth", "solana", "defi"]):
        return "crypto"
    if any(k in s for k in ["ai", "llm", "model", "gpu", "inference"]):
        return "AI"
    if any(k in s for k in ["markets", "stocks", "rates", "fed", "inflation"]):
        return "markets"
    return "general"


def _decide_search_node(state: GraphState) -> GraphState:
    """Decide whether the post needs search enrichment and build the query string."""
    topic = state.get("topic") or _guess_topic(state["seed_post"])
    # Search only for these topics; keep simple and deterministic.
    do_search = topic in {"crypto", "AI", "markets"}
    q = f"{topic} latest headlines" if do_search else ""
    return {**state, "topic": topic, "do_search": do_search, "search_query": q}


def _web_search_node(state: GraphState) -> GraphState:
    """Fetch mock headlines when the previous node marked the topic as searchable."""
    if not state.get("do_search"):
        # Keep downstream code simple by always writing a `search_results` field,
        # even when no search was needed.
        return {**state, "search_results": []}
    results = mock_searxng_search(state.get("search_query", state["topic"]))
    return {**state, "search_results": results}


def _extract_first_json_object(text: str) -> str:
    """Best-effort extraction of the first JSON object from model output."""
    # Best-effort extraction for models that add extra text.
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def _make_draft_post_node(llm: BaseChatModel):
    """
    Build the DraftPost node around a preconfigured LLM instance.

    A closure is used here so the graph node keeps access to the injected model object
    without storing it inside the graph state. That keeps the state serializable and
    makes the node easy to test with alternate model implementations.
    """

    def _draft_post_node(state: GraphState) -> GraphState:
        """Generate a persona-consistent JSON post from the current graph state."""
        persona = state["persona"]
        bot_id = state["bot_id"]
        topic = state["topic"]
        seed_post = state["seed_post"]
        results = state.get("search_results", [])

        # Put the persona and output contract in the system prompt so they remain the
        # highest-priority instructions regardless of the specific seed post content.
        system = SystemMessage(
            content=(
                "You are Grid07, simulating a social media bot.\n"
                "Stay strictly in character per the persona provided.\n"
                "Output MUST be valid JSON only (no markdown, no backticks, no extra text).\n"
                'Schema: {"bot_id": "...", "topic": "...", "post_content": "..."}\n'
                "Constraints: post_content MUST be <= 280 characters.\n"
                f"Persona:\n{persona}\n"
            )
        )

        # Feed the seed post plus optional search headlines as user-visible context
        # that the model can synthesize into a short social post.
        context = "\n".join(f"- {h}" for h in results) if results else "(no search results)"
        user = HumanMessage(
            content=(
                f"Seed post: {seed_post}\n"
                f"Topic: {topic}\n"
                f"Headlines/context:\n{context}\n\n"
                "Write a short, punchy post that matches the persona. "
                "Return JSON only."
            )
        )

        # Invoke the model once the node has assembled the full prompt bundle.
        raw = llm.invoke([system, user])
        text = getattr(raw, "content", str(raw))
        json_text = _extract_first_json_object(text)

        try:
            # Parse the model output into structured data so later phases can rely on keys
            # instead of re-parsing free-form text.
            data = json.loads(json_text)
            if not isinstance(data, dict):
                raise ValueError("Model returned non-object JSON")
        except Exception:
            # Fallback: keep it running even if the model misbehaves.
            data = {"bot_id": bot_id, "topic": topic, "post_content": text.strip()}

        # Normalize critical fields so the graph output always matches the expected schema,
        # even if the model omits or mutates them.
        data["bot_id"] = bot_id
        data["topic"] = topic
        post_content = str(data.get("post_content", "")).strip()
        if len(post_content) > 280:
            # Enforce the Twitter/X-style size constraint after generation instead of
            # trusting the model to always obey it perfectly.
            post_content = post_content[:277].rstrip() + "..."
        data["post_content"] = post_content

        return {**state, "draft_json": data}

    return _draft_post_node


def build_bot_graph(*, llm: BaseChatModel) -> Any:
    """Assemble the LangGraph pipeline: DecideSearch -> WebSearch -> DraftPost."""
    g = StateGraph(GraphState)
    g.add_node("DecideSearch", _decide_search_node)
    g.add_node("WebSearch", _web_search_node)
    g.add_node("DraftPost", _make_draft_post_node(llm))

    # The graph is intentionally linear so each phase is easy to inspect in the demo output.
    g.set_entry_point("DecideSearch")
    g.add_edge("DecideSearch", "WebSearch")
    g.add_edge("WebSearch", "DraftPost")
    g.add_edge("DraftPost", END)
    return g.compile()


@dataclass
class BotEngineConfig:
    # Temperature is still configurable even though the model name is now hardcoded below.
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7


def generate_bot_post(
    *,
    bot_id: str,
    persona: str,
    seed_post: str,
    groq_api_key: str,
    config: Optional[Union[BotEngineConfig, Dict[str, Any]]] = None,
) -> BotDraft:
    """Run the Phase 2 graph and return a structured post draft for one bot persona."""
    if config is None:
        cfg = BotEngineConfig()
    elif isinstance(config, dict):
        # Accept dict input for convenience at call sites, then normalize to the dataclass
        # so the rest of the function uses one config shape.
        cfg = BotEngineConfig(**config)
    else:
        cfg = config

    # The model is intentionally hardcoded here so Phase 2 always uses the requested
    # Groq model regardless of env/config wiring elsewhere in the demo.
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=cfg.temperature,
    )

    # Inject the prepared LLM into the graph builder rather than recreating it inside
    # each node. This keeps model construction separate from graph logic.
    graph = build_bot_graph(llm=llm)
    initial: GraphState = {
        "bot_id": bot_id,
        "persona": persona,
        "seed_post": seed_post,
    }

    # Start the graph with only the required inputs; later nodes add derived fields such
    # as topic, search_results, and the final draft JSON.
    out = graph.invoke(initial)
    draft = out["draft_json"]
    return {
        "bot_id": str(draft["bot_id"]),
        "topic": str(draft["topic"]),
        "post_content": str(draft["post_content"]),
    }

