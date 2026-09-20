# Agent layer

The live LangGraph agent and Streamlit UI. See the top-level README.md for the overview, architecture, setup and environment variables.

## Files

- `state.py` -- LangGraph state schema.
- `tools.py` -- tool implementations (Pinecone search, Mem0 read and write, closet persistence) and the LLM calls (vision, query building, ranking and styling note).
- `graph.py` -- graph wiring: nodes, conditional edges, retry-once logic and the human-approval interrupts.
- `demo.py` -- eight end-to-end scenarios that exercise every branch (needs live API keys).
- `streamlit_app.py` -- three-page UI (Upload & Recommend, My Closet, Preferences) with a profile switcher.
- `.streamlit/config.toml` -- Soft Autumn theme; only picked up when Streamlit is launched from this directory.
- `test_images/` -- photos used by the demo scenarios.
- `data/closets/` and `uploads/` -- runtime data, git-ignored.

## Running (from this directory)

```bash
streamlit run streamlit_app.py --server.port=8501
python demo.py
```
