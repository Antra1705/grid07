# Execution Logs

## Phase 1

```text
/Users/antrasingh/Desktop/grid07/phase1_router.py:84: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
  self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted: '/Users/antrasingh/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/.no_exist/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/adapter_config.json'

==============================
GRID07 — AI Bot Simulation Demo
==============================

### PHASE 1 — Vector Persona Router
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 7731.64it/s]
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted: '/Users/antrasingh/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/.no_exist/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/processor_config.json'
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted: '/Users/antrasingh/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/.no_exist/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/preprocessor_config.json'
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted: '/Users/antrasingh/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/.no_exist/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/video_preprocessor_config.json'
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted: '/Users/antrasingh/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/.no_exist/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/preprocessor_config.json'
- similarity[tech_maximalist] = 0.5364
- similarity[doomer_skeptic] = 0.3144
- similarity[finance_bro] = 0.2632

Matched bots (threshold=0.3): ['tech_maximalist', 'doomer_skeptic']
```

## Phase 2

```text
### PHASE 2 — LangGraph Content Engine
- Draft from `tech_maximalist` (<=280 chars, JSON): {'bot_id': 'tech_maximalist', 'topic': 'crypto', 'post_content': 'AI & crypto will revolutionize finance! ETF inflows surging, DeFi volumes rebounding. Regulatory noise is just that - noise. Bullish on Bitcoin and Ethereum!'}
- Draft from `doomer_skeptic` (<=280 chars, JSON): {'bot_id': 'doomer_skeptic', 'topic': 'crypto', 'post_content': "Crypto 'rebound' just a distraction from the AI-powered oligarchy eating our economy. Don't get duped, the system is rigged."}
```

## Phase 3

```text
### PHASE 3 — RAG Defense Engine
- Bot `tech_maximalist` defense reply:
The naysayers are out in full force, but I'm not buying it. Regulatory concerns are overblown, and the fundamentals of crypto are stronger than ever. Bitcoin and Ethereum are poised for a massive breakout, and the surge in DeFi volumes is just the beginning. Buckle up, because the future of finance is being written, and it's going to be a wild ride! Elon Musk is already making moves, and space-based crypto transactions are on the horizon. The bubble talk is just FUD - we're still in the early days of a revolution that will change the world.
```
