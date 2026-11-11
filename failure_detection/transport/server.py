from __future__ import annotations

import uuid
import logging
import time
from contextlib import asynccontextmanager
from threading import Lock
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..pipeline import Frame, SceneSnapshot, StreamObservation, StreamingRoboSentinel, Task
from .codec import decode_jpeg, json_safe
from .factory import load_pipeline_factory
from .result_store import ResultStore


class CreateTaskRequest(BaseModel):
    task_id: str
    description: str
    timestamp_sec: float
    frames: dict[str, str]


@dataclass
class Session:
    pipeline: StreamingRoboSentinel
    store: ResultStore
    next_frame_id: int = 0
    connected: bool = False


def _frames(values: dict[str, str], timestamp: float, frame_id: int) -> dict[str, Frame]:
    required = {"external", "wrist"}
    if required - set(values):
        raise ValueError("Both external and wrist frames are required")
    return {
        camera: Frame(camera, timestamp, frame_id, decode_jpeg(values[camera]))
        for camera in required
    }


def create_app(config: dict[str, Any]) -> FastAPI:
    model_status = {"models_ready": False, "resident_models": []}

    @asynccontextmanager
    async def lifespan(app):
        preload = config.get("deployment", {}).get("preload_factory")
        if preload:
            logging.getLogger(__name__).warning("Loading three models onto CUDA; wait for READY")
            details = await run_in_threadpool(load_pipeline_factory(preload), config)
            model_status.update(details)
            model_status["models_ready"] = True
            logging.getLogger(__name__).warning("READY: %s", model_status)
        yield
        sessions.clear()

    app = FastAPI(title="RoboSentinel GPU Service", lifespan=lifespan)
    sessions: dict[str, Session] = {}
    creation_lock = Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_factory_configured": bool(config.get("deployment", {}).get("pipeline_factory")),
            "active_tasks": len(sessions),
            **model_status,
        }

    def build_task(request: CreateTaskRequest) -> dict[str, Any]:
        if not creation_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another task is initializing")
        try:
            if len(sessions) >= int(config.get("deployment", {}).get("max_tasks", 1)):
                raise HTTPException(status_code=409, detail="Task limit reached; delete the existing task first")
            frames = _frames(request.frames, request.timestamp_sec, 0)
            factory = load_pipeline_factory(config.get("deployment", {}).get("pipeline_factory"))
            pipeline = factory(config)
            pipeline.initialize(
                Task(request.task_id, request.description),
                SceneSnapshot(request.timestamp_sec, frames),
            )
            session_id = uuid.uuid4().hex
            store = ResultStore(config, session_id)
            initial_perception = getattr(getattr(pipeline.tracker, "backend", None), "initial_perception", None)
            initial = {
                "record_type": "initialized", "task_session_id": session_id,
                "task_id": request.task_id, "description": request.description,
                "frame_id": 0, "timestamp_sec": request.timestamp_sec,
                "tracking_plan": asdict(pipeline.plan) if is_dataclass(pipeline.plan) else None,
                "rule_package": asdict(pipeline.rule_package)
                if is_dataclass(getattr(pipeline, "rule_package", None)) else None,
                "perception": asdict(initial_perception) if is_dataclass(initial_perception) else None,
            }
            initial = store.write(json_safe(initial), frames)
            sessions[session_id] = Session(pipeline, store, 1)
            return {"task_session_id": session_id, "initial_result": initial}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            creation_lock.release()

    @app.post("/tasks")
    async def create_task(request: CreateTaskRequest, connection: Request) -> dict[str, Any]:
        result = await run_in_threadpool(build_task, request)
        if await connection.is_disconnected():
            sessions.pop(result["task_session_id"], None)
        return result

    @app.websocket("/tasks/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="Unknown task session")
            return
        if session.connected:
            await websocket.close(code=4409, reason="Task already has an active stream")
            return
        session.connected = True
        await websocket.accept()
        try:
            while True:
                value = await websocket.receive_json()
                timestamp = float(value["timestamp_sec"])
                frames = _frames(value["frames"], timestamp, session.next_frame_id)
                session.next_frame_id += 1
                started = time.perf_counter()
                manual_failure = value.get("manual_failure", False)
                if not isinstance(manual_failure, bool):
                    raise ValueError("manual_failure must be boolean")
                update = await run_in_threadpool(session.pipeline.update, StreamObservation(
                    timestamp, frames, value.get("robot_state"), value.get("synchronization", {}), manual_failure
                ))
                result = json_safe(update.to_dict())
                result.update(record_type="update", task_session_id=session_id, frame_id=session.next_frame_id - 1)
                result["robot_state_received"] = value.get("robot_state") is not None
                result["manual_failure"] = manual_failure
                result["server_processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
                result = await run_in_threadpool(session.store.write, result, frames)
                await websocket.send_json(result)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        finally:
            sessions.pop(session_id, None)

    @app.delete("/tasks/{session_id}")
    def delete_task(session_id: str) -> dict[str, bool]:
        session = sessions.get(session_id)
        if session is not None and session.connected:
            raise HTTPException(status_code=409, detail="Close the active stream before deleting its task")
        return {"deleted": sessions.pop(session_id, None) is not None}

    return app
