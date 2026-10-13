from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..pipeline import Frame, SceneSnapshot, StreamObservation, StreamingRoboSentinel, Task
from .codec import decode_jpeg, json_safe
from .factory import load_pipeline_factory


class CreateTaskRequest(BaseModel):
    task_id: str
    description: str
    timestamp_sec: float
    frames: dict[str, str]


@dataclass
class Session:
    pipeline: StreamingRoboSentinel
    next_frame_id: int = 0


def _frames(values: dict[str, str], timestamp: float, frame_id: int) -> dict[str, Frame]:
    required = {"external", "wrist"}
    if required - set(values):
        raise ValueError("Both external and wrist frames are required")
    return {
        camera: Frame(camera, timestamp, frame_id, decode_jpeg(values[camera]))
        for camera in required
    }


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="RoboSentinel GPU Service")
    sessions: dict[str, Session] = {}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_factory_configured": bool(config.get("deployment", {}).get("pipeline_factory")),
            "active_tasks": len(sessions),
        }

    @app.post("/tasks")
    def create_task(request: CreateTaskRequest) -> dict[str, str]:
        try:
            factory = load_pipeline_factory(config.get("deployment", {}).get("pipeline_factory"))
            pipeline = factory(config)
            frames = _frames(request.frames, request.timestamp_sec, 0)
            pipeline.initialize(
                Task(request.task_id, request.description),
                SceneSnapshot(request.timestamp_sec, frames),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        session_id = uuid.uuid4().hex
        sessions[session_id] = Session(pipeline, 1)
        return {"task_session_id": session_id}

    @app.websocket("/tasks/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="Unknown task session")
            return
        await websocket.accept()
        try:
            while True:
                value = await websocket.receive_json()
                timestamp = float(value["timestamp_sec"])
                frames = _frames(value["frames"], timestamp, session.next_frame_id)
                session.next_frame_id += 1
                update = session.pipeline.update(StreamObservation(
                    timestamp, frames, value.get("robot_state"), value.get("synchronization", {})
                ))
                await websocket.send_json(json_safe(update.to_dict()))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)

    @app.delete("/tasks/{session_id}")
    def delete_task(session_id: str) -> dict[str, bool]:
        return {"deleted": sessions.pop(session_id, None) is not None}

    return app
