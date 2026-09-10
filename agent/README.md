# Digital Closet Agent — LangGraph Skeleton

This is the agentic control-flow layer described in Section 6 of
`Project_Master_Agentic_Context_Ledger.md`, built and runnable now, ahead of
the embedding/Pinecone pipeline. It satisfies the Week 3 handout's core bar
("decides what to do next, calls tools, holds state across steps, recovers
from errors, and hands off to a human") independently of whether retrieval
is backed by stubs or real Pinecone indexes — the graph shape doesn't change
when you swap those in.

## Files

- `state.py` — the session-state schema (`ClosetAgentState`). Maps to
  Section 6's "What does it need to remember?" row.
- `tools.py` — every tool call (`search_brand_inventory`,
  `get_user_preferences`, `save_to_digital_closet`, plus the vision and
  orchestration LLM calls), all currently **stubbed**. Each function has a
  `REAL IMPLEMENTATION` comment block telling you exactly what to swap in.
- `graph.py` — the actual LangGraph `StateGraph`: nodes, conditional edges,
  retry logic, and the human-approval interrupt gate.
- `demo.py` — runs three scenarios (happy path, low-confidence re-upload,
  search failure + retry) so every branch executes at least once. This is
  a good basis for the video demo — run it live and narrate the trace.

## Running it

```bash
uv pip install langgraph langchain-core   # or plain pip
python demo.py
```

## What's real vs. stubbed right now

| Piece | Status |
|---|---|
| Graph shape, conditional routing, retry-once logic, human-approval gate | **Real** |
| `vision_extract_attributes` | Stub — swap in your GLM-5.3-Flash call (see `backfill_attributes.py`'s `call_vision_model*` for the request shape) |
| `search_brand_inventory` | Stub — swap in the dual Pinecone query (bge-m3 + CLIP) once the embedding pipeline is live |
| `get_user_preferences` | Stub — swap in the real Mem0 fetch |
| `save_to_digital_closet` | Stub — swap in the PostgreSQL MCP write |
| `build_complementary_query` / `rank_and_style` | Stub — swap in the Llama-3.3-70B (Nebius) calls |

**None of these swaps require touching `graph.py`.** The stub functions in
`tools.py` are the seam — as long as the real implementation keeps the same
function signature and return shape, the graph doesn't change.

## How this maps to the handout's grading criteria

| Handout asks for | Where it lives here |
|---|---|
| Decides what to do next | `route_after_vision`, `route_after_search` conditional edges |
| Holds state across steps | `ClosetAgentState`, threaded through via `MemorySaver` checkpointer + `thread_id` |
| Calls tools | `tools.py` functions, invoked from graph nodes |
| Recovers from errors | `node_search_brand_inventory`'s retry-once-then-message logic |
| Hands off to a human | `interrupt_before=["save_to_digital_closet"]` — the graph physically cannot execute the write without an external `update_state` call setting `user_wants_to_save` |

## Next steps (suggested order, given the Sept 16 deadline)

1. Wire `search_brand_inventory` to a real (even single-index, text-only)
   Pinecone query — this unblocks a genuinely live demo sooner than waiting
   for both bge-m3 and CLIP to be ready.
2. Wire `vision_extract_attributes` to GLM-5.3-Flash.
3. Wire `get_user_preferences` / `save_to_digital_closet` to Mem0 / Postgres.
4. Put a thin Streamlit UI in front of `build_graph()` for the demo video.
5. Add `build_complementary_query` / `rank_and_style` LLM calls last — they
   matter for recommendation *quality*, not for satisfying the agentic bar.
