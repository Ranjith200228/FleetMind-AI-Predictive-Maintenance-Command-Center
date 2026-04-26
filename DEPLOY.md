# Deploying FleetMind to Hugging Face Spaces

This is a runbook for publishing the dashboard at
`https://huggingface.co/spaces/Ranjithmaddirala/fleetmind-ai`.

## What ships in the Space

| Path                              | Size  | Purpose                           |
| --------------------------------- | ----- | --------------------------------- |
| `app.py`                          | ~12 K | Streamlit entry point             |
| `src/`                            | ~70 K | LSTM, RAG, agent, viz, helpers    |
| `models/lstm_fd001.keras`         | 1.5 M | Trained 2-layer LSTM              |
| `models/preprocess_fd001.joblib`  | 1.5 K | Fitted MinMax scaler + meta       |
| `data/raw/{train,test,RUL}_FD001` | 5.6 M | C-MAPSS FD001 (public NASA PCoE)  |
| `data/rag_index/`                 | 437 K | FAISS index + 55 doc chunks       |
| `data/rag/{corpus,eval_qa}.jsonl` | ~30 K | Source corpus + 62-pair eval set  |
| `reports/*.json`                  | ~2 K  | Metrics shown in the KPI bar      |
| `requirements.txt`                | <1 K  | UTF-8, exact deps                 |
| `.streamlit/config.toml`          | <1 K  | Dark theme + accent red           |
| `README.md`                       | ~10 K | HF frontmatter + project write-up |

The 48 MB Random Forest baseline (`models/rf_fd001.joblib`) is **not** shipped —
the app never loads it; its metrics live in `reports/metrics.json`.

## One-time setup

1. Create the Space (via the website or CLI):

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login           # paste a write token
   huggingface-cli repo create \
       fleetmind-ai \
       --type space \
       --space_sdk streamlit \
       --organization Ranjithmaddirala
   ```

2. Add the OpenAI key as a Space secret (Settings → Variables and secrets):

   ```
   OPENAI_API_KEY = sk-...
   ```

   Without it the agent silently falls back to the deterministic mock and the
   Copilot tab shows a yellow banner; with it you get real `gpt-4o-mini`
   tool-calling and real `text-embedding-3-small` retrieval.

## Push the code

From this worktree:

```bash
git remote add space https://huggingface.co/spaces/Ranjithmaddirala/fleetmind-ai
git push space HEAD:main
```

If you prefer a clean publish branch, push from a fresh clone instead of the
worktree to avoid carrying `.claude/`.

## Build behaviour

Hugging Face provisions ~2 vCPU / 16 GB RAM on the free CPU tier. With our
`requirements.txt` the cold-build takes 4–6 min (TensorFlow 2.20 wheel is the
slow part). The first request after build then takes ~8 s for the LSTM to load
into memory; every subsequent request is sub-second thanks to
`st.cache_resource` on the model, scaler, retriever, and agent.

## Smoke-testing post-deploy

1. KPI bar shows `Fleet size 100`, `LSTM test RMSE 14.31`, `Retrieval hit@1 0.92`.
2. Fleet Overview tab — heatmap renders, REPLACE/REPAIR counts > 0.
3. Engine Detail tab — switch to engine 100, the 3D model recolours and the
   health-by-stage panel shows HPC z > 3.
4. Copilot tab — click *"What's the current RUL for engine 17 and what should
   we do?"*. Expect a tool call to `query_engine_history` with `engine_id=17`
   and at least one `[doc_id]` citation.

## Updating the Space

```bash
git push space HEAD:main           # pushes new commits
```

The Space rebuilds automatically.
