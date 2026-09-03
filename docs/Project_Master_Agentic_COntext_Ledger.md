### Project State & Context Ledger

### 1. Project Overview
* Name: Digital Closet / Style Aggregator (referred to interchangeably as "Kiran's Digital Closet")
* Objective: A personalized AI stylist aggregator that analyzes a user's existing closet items via photo upload and recommends real, purchasable matching items from partner brands — retrieval-based, not image generation. This project also serves as the Week 3 "Build Your AI Agent" submission for The Gen Academy (certification track, due Sept 16, 2026), so it must satisfy an agentic-system bar (decision-making, tool use, state, error recovery, human handoff) — not just a retrieval pipeline.
* Target Audience: Not explicitly discussed in this thread. Worth defining before further build work — even a rough persona (e.g., "budget-conscious 20s-30s shopper who wants outfit-matching without browsing multiple brand sites") would sharpen tool design and success metrics in Section 6.

### 2. Tech Stack & Environment
Languages: Python (confirmed — requests, dataclasses used in the scraper; no other language discussed).
  ### Frameworks:
- LlamaIndex — binds Pinecone vector search and Mem0 user profiles together (retrieval/indexing layer).
- LangChain + LangGraph — new addition per Week 3 requirement. LangGraph becomes the stateful control-flow layer sitting above LlamaIndex/Pinecone: it owns session state, decides which tool to call next, branches on low-confidence or failed results, and manages retries. LlamaIndex is not replaced — it stays as the retrieval mechanism the graph calls into.
- No frontend framework discussed yet.

  ### Database/Storage:
- Pinecone (Serverless) — two separate indexes/namespaces: bge-m3 for text embeddings, CLIP for image embeddings (confirmed non-interchangeable, must stay separate).
- PostgreSQL — user tables, auth data, digital closet items, via a PostgreSQL MCP server. Not yet built out.
  
  ### Key Libraries: requests (scraper HTTP calls). Embedding/Pinecone SDK scripts, LangGraph, and Mem0 client libraries are pending — none installed/confirmed yet.

### 3. Core Architecture & Rules
  ### Design Patterns: Two-part architecture:
- Offline ingestion pipeline (unchanged): scrape → embed (bge-m3 text / CLIP image, parallel branches) → index (two Pinecone namespaces).
- Runtime pipeline, now agentic rather than fixed-sequence: photo upload → vision-tag (Qwen2.5-VL) → dual Pinecone retrieval → orchestration (Llama-3.3-70B) → display, wrapped in a LangGraph state machine that makes real decisions at each step rather than always proceeding linearly (e.g., re-prompt the user on a low-confidence vision tag; broaden the query if Pinecone returns too few matches).
- Agent/tool-calling layer via MCP: search_brand_inventory (read), get_user_preferences (read), save_to_digital_closet (write).

  ### Hard Constraints:
- No image-generation LLM — all recommendations must be real, retrievable partner-brand products.
- Text and image embeddings must be stored and queried in separate vector spaces/indexes (bge-m3 and CLIP are not interchangeable or comparable).
- Data sourcing limited to free/sanctioned methods for MVP — public Shopify /products.json endpoints preferred over paid scraping services or brand APIs that don't exist (Zara/H&M/Banana Republic have no public dev API).
- New (Week 3 requirement): write actions require human approval by default. save_to_digital_closet must prompt for user confirmation before executing — it creates/modifies data. search_brand_inventory and get_user_preferences are reads and remain autonomous.
- New: defined failure handling is required, not optional. Every tool call needs an explicit behavior for the "it broke" case (see Section 6 for the specific rule) rather than being left undefined.

  ### Style Preferences: Not explicitly discussed in this thread — no formatting/linting/style-guide conventions specified.


### 4. Current State & Milestones

* **Completed:** 

  * [Need to refactor API authentication]
  ### Completed:
- Corrected embedding architecture: confirmed bge-m3 (text) and CLIP (image) are separate, non-comparable vector spaces — pipeline redesigned to use two parallel branches meeting at query time.
- Evaluated brand data sourcing options and identified the public, unauthenticated Shopify /products.json endpoint as the free MVP data source, based on analysis of the reference dataset (KiransDigitalCloset-Preview.html).
- Built and delivered shopify_scraper.py and validate_domains.py — domain validation confirmed all 7 target stores (SABO, Petal & Pup, Meshki, Bohme, Natural Life, Oh Polly, Red Dress) expose a working /products.json endpoint.
Built and delivered shopify_scraper.py and validate_domains.py — domain validation confirmed all 7 target stores (SABO, Petal & Pup, Meshki, Bohme, Natural Life, Oh Polly, Red Dress) expose a working /products.json endpoint.
- Produced two finalized architecture diagrams: offline ingestion pipeline, and runtime query pipeline (photo upload → vision tag → dual Pinecone query → orchestration LLM with Mem0 → frontend cards).
- Evaluated Kimi K3 (Moonshot AI, via Fireworks) as a potential model swap-in for vision tagging and/or orchestration; decided against including it in v1.
- Reviewed Week 3 Project Handout requirements against existing architecture; identified the gap between the current design (RAG pipeline) and the handout's agentic-system bar; decided to pursue the certification-only deadline (Sept 16) rather than the Builder of the Week deadline (Aug 30).

  ### In Progress:
- Running shopify_scraper.py against the validated 7-domain list to generate seed_products.json.
- Drafting the Agent Framework document (one-liner + 9-row table — see Section 6) for the Week 3 submission.

  ### Blockers / Technical Debt:
- No blockers currently. Note: full LangGraph agent layer is intentionally sequenced after the embedding/Pinecone/orchestration build (Section 6 build order) rather than built first — this is a deliberate sequencing choice, not a blocker.


  ### 🚧 Updated In-Progress / Next Steps (Add to Section 4):


    * [Building Feature B]
* **Blockers / Technical Debt:**

### 5. Active Decisions & Context
- August 27, 2026: Adopted the public Shopify /products.json endpoint as the primary free data source for MVP seed data — no open developer API exists for Zara, H&M, or Banana Republic; this endpoint is free, requires no approval, and covers the bulk of the reference dataset's item count.
- August 27, 2026: Split the vector storage layer into two separate Pinecone indexes/namespaces (text via bge-m3, image via CLIP) rather than one combined index — the two embedding models produce non-comparable vector spaces.
- August 27, 2026: Kept Qwen2.5-VL (attribute tagging) and Llama-3.3-70B (orchestration) as v1 model choices; deferred Kimi K3 to a possible v2 upgrade for the orchestration layer only, due to its higher cost/latency profile relative to v1's narrow structured-output tasks.
- August 29, 2026: Targeting the Sept 16, 2026 certification-only deadline for Week 3, not the Aug 30 Builder of the Week deadline — chosen to allow the full pipeline (scraper → dual embeddings → Pinecone → Mem0 → Vision/Orchestration LLMs) to be built properly before layering in the LangGraph agent control, rather than rushing a scoped-down agent to meet the earlier date.
- August 29, 2026: Adopted LangChain + LangGraph as the agentic control layer, added on top of the existing LlamaIndex/Pinecone retrieval stack rather than replacing it — LangGraph owns state, branching decisions, and retries; LlamaIndex remains the retrieval mechanism it calls into.
- August 29, 2026: Established that save_to_digital_closet requires human approval before executing (write action), while search_brand_inventory and get_user_preferences remain autonomous (read actions) — per the handout's rule that write actions default to human review.

### 6. Agent Framework (new section — Week 3 requirement)

- One-liner: My agent helps me do upload my current closet items and search for matching casual, formal different matching suitable dresses(bottom or top or accessories) in web app(later mobile app), replacing hours of manual cross-brand browsing it takes to style an existing closet item today by browsing  all my favorite brands in one place. It extracts style attributes from an uploaded photo and retrieves real matching products on its own using 3 tools, hands off to the user before saving anything to their closet, and I'll know it works when a user can get a usable outfit recommendation in under 30 seconds that they'd actually click through to buy.

Field	Fill in
- Agent goal	Takes a photo of a user's closet item and returns real, purchasable matching pieces with styling advice.
- Where do people use it?	Web app (frontend cards displaying recommendations with links to brand product pages).
- What steps does it take, in order?	1) Receive photo upload. 2) Vision LLM extracts JSON attributes. 3) Query both Pinecone indexes (CLIP image similarity + bge-m3 text similarity) via search_brand_inventory. 4) Fetch user profile via get_user_preferences (Mem0). 5) Orchestration LLM ranks candidates and writes styling advice. 6) Present cards to user. 7) On user request, save_to_digital_closet — pending approval.
- What can it actually do?	search_brand_inventory (read — queries Pinecone), get_user_preferences (read — queries Mem0), save_to_digital_closet (write — requires human approval).
- What does it need to remember?	Session: current upload, extracted attributes, retrieved candidates. Persistent (via Mem0): user style preferences, disliked colors/brands, saved closet items across sessions.
- What should it never do?	Never save an item to the user's digital closet without explicit confirmation. Never recommend a product without a live, retrievable product_url. Never fabricate product data not present in the Pinecone index.
- Human-in-the-loop	Before save_to_digital_closet executes (write action). User can also override/reject the vision LLM's extracted attributes before retrieval proceeds.
- What happens when something breaks?	Vision LLM low-confidence output → ask user to re-upload rather than proceed. Pinecone returns too few matches → broaden query once, then tell the user no strong match was found rather than fail silently. Tool call errors (timeout, dead link) → retry once, then surface a plain-language message to the user.
- How do you know it worked?	Target: usable, clickable recommendation returned for at least 8 out of 10 test uploads. (Placeholder metric — refine once real test uploads are run.)