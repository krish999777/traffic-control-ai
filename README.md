# Traffic Control AI

A local, multi-agent traffic analysis pipeline. A camera frame goes in, a vision model reads the scene, a decision agent (backed by RAG over historical cases) judges whether the situation needs intervention, and a solution agent (also backed by RAG) proposes a fix grounded in what's worked before. A human reviews and rates the result, and good ratings feed back into the retrieval store for future runs.

Everything runs locally through [Ollama](https://ollama.com), no cloud API calls, no inference cost.

## How it works

```
Frame in
  ↓
Vision Analysis (structured output: vehicle counts, density, order, observation)
  ↓
Decision Similarity Search (RAG: retrieve similar past cases)
  ↓
Decision Agent (is a solution actually needed?)
  ↓
  ├─ No  → END ("No solution needed")
  └─ Yes → Solution Similarity Search (RAG: retrieve similar past cases + prior ratings)
              ↓
           Solution Agent (propose a fix grounded in retrieved context)
              ↓
           Human Review (interrupt / resume)
              ↓
           Rating ≥ 3 → embedded into the solution vector store for future retrieval
           Rating ≤ 2 → discarded
              ↓
             END
```

The decision agent is specifically tuned to avoid a common false positive: a long line of vehicles waiting in an orderly queue at a traffic signal is not, by itself, a traffic problem. The vision model's structured output distinguishes traffic *density* from traffic *order*, and the decision agent uses both, along with retrieved historical cases, to make that call.

The feedback loop is not a self-training system. It's a growing, human-curated case library: every well-rated solution becomes future context for the retrieval step, so later decisions and solutions are informed by real prior outcomes rather than the model's assumptions alone.

## Screenshots
<img width="1470" height="801" alt="traffic control ai post 1" src="https://github.com/user-attachments/assets/5cf8db48-1c04-4976-9f83-a9b9ea35039d" />
<img width="1464" height="801" alt="Screenshot 2026-08-17 at 5 50 57 PM" src="https://github.com/user-attachments/assets/359917d8-8c14-4095-afc1-1134a8295e32" />
<img width="286" height="630" alt="graph" src="https://github.com/user-attachments/assets/225f6181-6b1a-4885-bb05-a76f80dd27ee" />
<img width="986" height="313" alt="Screenshot 2026-08-17 at 5 40 23 PM" src="https://github.com/user-attachments/assets/301c1b89-96df-48f4-b655-27b1fd9697d9" />

## Stack

- **Ollama** — local model runtime (vision, text, and embedding models)
- **LangChain** — model wrappers, structured output, retrieval
- **LangGraph** — graph orchestration, conditional routing, `interrupt`/`resume` for human-in-the-loop, checkpointing via `MemorySaver`
- **Qdrant** — vector store for the decision and solution case libraries
- **Express** — backend API
- **React (Vite)** — frontend

## Architecture notes

- Vision analysis is a direct structured-output call, no retrieval involved, it's a perception task, not a retrieval task.
- Decision and solution steps each run their own similarity search against separate Qdrant collections before invoking their respective model, so retrieval context is scoped to what's actually relevant to that decision.
- Routing is handled via LangGraph's `Command`, the decision agent's own output determines whether the graph proceeds to the solution step or ends, no separate branching node.
- Human review uses a genuine `interrupt()`/`resume()` step backed by a checkpointer (`thread_id` per run), not a simulated pause.

## Setup

1. Install [Ollama](https://ollama.com) and pull the required models:
   ```bash
   ollama pull minicpm-v4.5:8b        # vision model
   ollama pull lfm2.5:8b              # decision / solution model
   ollama pull nomic-embed-text-v2-moe # embeddings
   ```
2. Set up a Qdrant instance (local or cloud) and create the two collections used for retrieval (`traffic-decision`, `traffic-solution`).
3. Add a `.env` file with:
   ```
   QDRANT_URL=your-qdrant-url
   QDRANT_API_KEY=your-qdrant-api-key
   ```
4. Install dependencies and run the backend:
   ```bash
   npm install
   npm run dev
   ```
5. Run the frontend (in the `frontend/` directory):
   ```bash
   npm install
   npm run dev
   ```

## API

- `POST /image` — upload a traffic frame (`multipart/form-data`, field name `traffic_frame`). Runs the full pipeline up to human review and returns a `thread_id` if a solution was proposed.
- `POST /feedback` — `{ rating: 1-5, thread_id }`. Resumes the interrupted graph run; ratings of 3+ embed the solution into the case library.
- `POST /decisionRag` / `POST /solutionRag` — manually seed the respective vector stores with historical cases.

## Limitations

- **Not deployed.** Every model runs locally through Ollama, which is the whole point of the setup, but it also means there's no hosted version to try without running it yourself.
- **No live camera integration.** Frames are supplied as image uploads, in testing these were extracted from real traffic footage at a fixed interval, not pulled from a live feed.
- **Decision case library is seeded manually** (via `/decisionRag`) rather than growing automatically from live pipeline runs, a deliberate scope decision to keep the project focused on demonstrating the LangGraph/RAG architecture rather than building a fully autonomous data-collection system.
