### Project State & Context Ledger

### 1. Project Overview

* Name: KiransDigitalCloset-Preview or Style_Aggregator
* Objective: A personalized AI stylist aggregator that analyzes a user's existing closet items via photo upload and recommends real, purchasable matching items from partner brands — retrieval-based, not image generation.
* Target Audience: Not explicitly discussed in this thread.

### 2. Tech Stack & Environment

Languages: Python (confirmed — used for the scraper script; requests library, dataclasses).
Frameworks: LlamaIndex (to bind Pinecone vector search and Mem0 user profiles together). No frontend framework was discussed in this thread.
Database/Storage:
Pinecone (Serverless) — two separate indexes/namespaces: one for bge-m3 text embeddings, one for CLIP image embeddings.
PostgreSQL (mentioned as the store for user tables, auth data, and digital closet items — via a PostgreSQL MCP server) — not yet built out in this thread.
Key Libraries: requests (scraper HTTP calls). No other libraries confirmed yet — embedding/Pinecone SDK scripts are still pending (next step).

### 3. Core Architecture & Rules

Design Patterns: Multi-stage pipeline architecture — offline ingestion (scrape → embed → index) fully decoupled from runtime query (upload → vision-tag → retrieve → orchestrate → display). Agent/tool-calling layer via MCP for search_brand_inventory, get_user_preferences, save_to_digital_closet.
Hard Constraints:
No image-generation LLM — all recommendations must be real, retrievable partner-brand products.
Text and image embeddings must be stored and queried in separate vector spaces/indexes (bge-m3 and CLIP are not interchangeable or comparable).
Data sourcing limited to free/sanctioned methods for MVP — public Shopify /products.json endpoints preferred over paid scraping services or brand APIs that don't exist (Zara/H&M/Banana Republic have no public dev API).
Style Preferences: Not explicitly discussed in this thread — no formatting/linting/style-guide conventions were specified.

### 4. Current State & Milestones

* **Completed:** 

  * [Need to refactor API authentication]
  ### 🆕 New Completions
- Corrected embedding architecture: confirmed bge-m3 (text) and CLIP (image) are separate, non-comparable vector spaces — pipeline redesigned to use two parallel branches meeting at query time, not a single merged index.
- Evaluated brand data sourcing options (direct brand APIs, affiliate networks, paid scraping aggregators) and identified the public, unauthenticated Shopify `/products.json` endpoint as the free MVP data source, based on analysis of the uploaded reference dataset (`KiransDigitalCloset-Preview.html`).
- Built and delivered `shopify_scraper.py` — pulls/paginates `/products.json` across a curated store list, normalizes into the target schema (brand, category, price, image_url, product_url, sizes, colors, tags), includes a domain-validation helper and rate limiting. Syntax-checked and saved to outputs.
- Produced two finalized architecture diagrams: (1) offline ingestion pipeline — scraper → bge-m3/CLIP embedding branches → two separate Pinecone indexes; (2) runtime query pipeline — photo upload → Vision LLM attribute extraction → dual Pinecone query → orchestration LLM (with Mem0 profile input) → frontend cards.
- Evaluated Kimi K3 (Moonshot AI, via Fireworks) as a potential model swap-in; decided against including it in v1.

### 🚧 Updated In-Progress / Next Steps (Add to Section 4):
- Run `shopify_scraper.py` against the validated store list (SABO, Petal & Pup, Meshki, Bohme, Natural Life, Oh Polly, Red Dress) to generate `seed_products.json`; spot-check sizes/colors mapping since a few stores may swap `option1`/`option2` conventions.
- Validate additional candidate Shopify store domains via `validate_domains()` before expanding `STORE_DOMAINS`.
- Write the CLIP embedding + Pinecone upsert script for the image index/namespace.
- Write the bge-m3 embedding + Pinecone upsert script for the text index/namespace.
- Decide on Replit deployment tier for the scraper specifically (reserved VM / Deployments, not free autoscale) if it needs to run on a schedule.
- Revisit Kimi K3 as a v2 candidate for the orchestration LLM role once real usage data exists, to weigh its agentic/tool-calling and native multimodal strengths against its higher cost vs. Llama-3.3-70B.

  * [Building Feature B]
* **Blockers / Technical Debt:** 

### 5. Active Decisions & Context

* **[Date]:** [Decision Name] - Why we chose X over Y (e.g., "Chose SQLite for local prototyping speed over Postgres").
### 💡 New Decisions & Context Ledger:
- **August 27, 2026:** Adopted the public Shopify `/products.json` endpoint as the primary free data source for MVP seed data, in place of brand APIs or affiliate feeds — Zara, H&M, and Banana Republic have no open developer API; this endpoint is free, requires no approval, and covers the bulk of the reference dataset's item count.
- **August 27, 2026:** Split the vector storage layer into two separate Pinecone indexes/namespaces (text via bge-m3, image via CLIP) rather than one combined index — the two embedding models produce non-comparable vector spaces, so results must be retrieved separately and merged at query time, not stored together.
- **August 27, 2026:** Kept Qwen2.5-VL (attribute tagging) and Llama-3.3-70B (orchestration) as the v1 model choices; deferred Kimi K3 to a possible v2 upgrade for the orchestration layer only, due to its higher cost/latency profile relative to the narrow, structured-output tasks in v1.