import os
import sys
import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator

app = FastAPI(title="HW4 Orchestrated Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# All available yearly CSVs
ALL_DATASETS = sorted([
    os.path.join(DATASETS_DIR, f)
    for f in os.listdir(DATASETS_DIR)
    if f.endswith(".csv")
])

# conversation_id -> session state
_sessions: dict[str, dict] = {}


def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    years: list[int] | None = None  # e.g. [2014, 2015] to filter datasets


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets():
    return {
        "datasets": [os.path.basename(p) for p in ALL_DATASETS],
        "years": sorted([
            int(os.path.basename(p).split("_")[0])
            for p in ALL_DATASETS
            if os.path.basename(p).split("_")[0].isdigit()
        ]),
    }


@app.post("/query")
async def query(request: QueryRequest):
    """
    Run the orchestrator and stream progress via Server-Sent Events.

    SSE event types:
      classify        — question routed to 'analytics' or 'web'
      web_search_start / web_search_done
      analytics_start / analytics_done  — per attempt
      validator_start / validator_done  — per attempt
      result          — final structured output (last event)
      error           — unhandled exception
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    session = _sessions.get(conversation_id, {})

    # Resolve dataset paths
    if request.years:
        dataset_paths = [
            p for p in ALL_DATASETS
            if any(str(y) in os.path.basename(p) for y in request.years)
        ]
    else:
        dataset_paths = ALL_DATASETS

    if not dataset_paths:
        raise HTTPException(status_code=400, detail="No matching dataset files found.")

    # Build conversation history context
    history = session.get("history", [])

    async def event_stream():
        events = []

        def on_event(event_type: str, payload: dict):
            events.append((event_type, payload))

        try:
            # We run synchronously and yield buffered events
            # (for a production system, use asyncio/threads)
            result = orchestrator.run(
                question=request.question,
                dataset_paths=dataset_paths,
                on_event=on_event,
            )

            # Yield all intermediate events
            for event_type, payload in events:
                yield _sse(event_type, payload)

            # Persist session
            history.append({
                "question": request.question,
                "answer": result["final_answer"],
                "source": result["source"],
            })
            _sessions[conversation_id] = {
                "history": history,
                "last_result": result,
            }

            # Final result event
            yield _sse("result", {
                "conversation_id": conversation_id,
                "question": request.question,
                "source": result["source"],
                "final_answer": result["final_answer"],
                "plots": result["plots"],
                "citations": result["citations"],
                "validator": result["validator"],
                "iterations": result["iterations"],
                "retries": result["retries"],
                "history": history,
            })

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{conversation_id}")
def get_history(conversation_id: str):
    session = _sessions.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "history": session.get("history", [])}
