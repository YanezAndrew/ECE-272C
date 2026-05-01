import os
import sys
import json
import uuid
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Project root is three levels up from backend/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from HW2.project.agent import agent, build_initial_state
from HW2.project.nodes import ExecutionResult

app = FastAPI(title="CSV Q&A Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# upload_id -> temp file path for user-uploaded CSVs
_uploaded_files: dict[str, str] = {}

# conversation_id -> persistent session state across turns
_sessions: dict[str, dict] = {}
# Each session:
# {
#   "csv_path": str,
#   "execution_result": ExecutionResult | None,
#   "history": [{"question": str, "answer": str}, ...]
# }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None  # omit to start a new conversation
    csv_path: str | None = None         # absolute server path
    upload_id: str | None = None        # from /upload
    dataset: str | None = None          # filename inside datasets/


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
    """Upload a CSV. Returns an upload_id to reference it in /query."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(contents)
    tmp.close()

    upload_id = str(uuid.uuid4())
    _uploaded_files[upload_id] = tmp.name
    return {"upload_id": upload_id, "filename": file.filename}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Run the agent and stream progress via Server-Sent Events.

    On the first turn omit conversation_id — the response includes a new one.
    Pass it back on every follow-up turn so the session is reused.

    SSE event types:
      node_done  – emitted after each LangGraph node completes
      result     – final structured output (last event, includes conversation_id)
      error      – unhandled exception
    """
    # --- resolve conversation ---
    conversation_id = request.conversation_id or str(uuid.uuid4())
    session = _sessions.get(conversation_id)

    # --- resolve CSV path ---
    if session:
        csv_path = session["csv_path"]
    else:
        csv_path = request.csv_path
        if not csv_path and request.upload_id:
            csv_path = _uploaded_files.get(request.upload_id)
        if not csv_path and request.dataset:
            csv_path = os.path.join(DATASETS_DIR, request.dataset)

    if not csv_path:
        raise HTTPException(
            status_code=400,
            detail="Provide csv_path, upload_id, or dataset on the first turn.",
        )
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    # --- build initial state, injecting previous result for follow-ups ---
    previous_result: ExecutionResult | None = session["execution_result"] if session else None
    initial_state = build_initial_state(request.question, csv_path, previous_result)

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

            new_result: ExecutionResult | None = final_state.get("execution_result")
            final_answer = final_state.get("final_answer", "")

            # --- persist session state ---
            history = session["history"] if session else []
            history.append({"question": request.question, "answer": final_answer})
            _sessions[conversation_id] = {
                "csv_path": csv_path,
                "execution_result": new_result,
                "history": history,
            }

            yield _sse("result", {
                "conversation_id": conversation_id,
                "question": request.question,
                "final_answer": final_answer,
                "generated_code": final_state.get("generated_code", ""),
                "execution_result": new_result.to_json() if new_result else None,
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
