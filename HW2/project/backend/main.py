import os
import sys
import json
import uuid
import tempfile

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Project root is three levels up from backend/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from HW2.project.agent import agent, build_initial_state

app = FastAPI(title="CSV Q&A Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

# session_id -> temp file path for user-uploaded CSVs
_uploaded_files: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_result(result) -> str | None:
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.to_json(orient="records")
    if isinstance(result, pd.Series):
        return result.to_json()
    return str(result)


def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    csv_path: str | None = None    # absolute server path
    session_id: str | None = None  # from /upload
    dataset: str | None = None     # filename inside datasets/ e.g. "housing.csv"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets():
    """List the built-in CSV files available in datasets/."""
    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
    return {"datasets": files}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV. Returns a session_id to reference it in /query."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(contents)
    tmp.close()

    session_id = str(uuid.uuid4())
    _uploaded_files[session_id] = tmp.name
    return {"session_id": session_id, "filename": file.filename}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Run the agent and stream progress via Server-Sent Events.

    Resolve CSV priority: csv_path > session_id > dataset name.

    SSE event types:
      node_done  – emitted after each LangGraph node completes
      result     – final structured output (last event)
      error      – unhandled exception
    """
    # Resolve which CSV to use
    csv_path = request.csv_path
    if not csv_path and request.session_id:
        csv_path = _uploaded_files.get(request.session_id)
    if not csv_path and request.dataset:
        csv_path = os.path.join(DATASETS_DIR, request.dataset)

    if not csv_path:
        raise HTTPException(
            status_code=400,
            detail="Provide csv_path, a session_id from /upload, or a dataset filename.",
        )
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    initial_state = build_initial_state(request.question, csv_path)

    async def event_stream():
        final_state = {}
        try:
            for chunk in agent.stream(initial_state):
                node_name = next(iter(chunk))
                node_state = chunk[node_name]
                final_state.update(node_state)

                yield _sse("node_done", {
                    "node": node_name,
                    "evaluation": node_state.get("evaluation"),
                    "retry_count": node_state.get("retry_count"),
                })

            yield _sse("result", {
                "question": request.question,
                "final_answer": final_state.get("final_answer", ""),
                "generated_code": final_state.get("generated_code", ""),
                "execution_result": _serialize_result(final_state.get("execution_result")),
                "evaluation": final_state.get("evaluation", "FAIL"),
                "visualization": {
                    "enabled": final_state.get("visualization_decision", False),
                    "chart_type": final_state.get("visualization_chart_type", "none"),
                    "figure_json": final_state.get("visualization_figure"),
                    "error": final_state.get("visualization_error"),
                },
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
