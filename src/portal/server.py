"""Paper-trading lab: a local web portal for running replay and live experiments.

Served by uvicorn inside the engine's own event loop, so live experiments see
every feature update with no inter-process hop. Binds to 127.0.0.1 by default:
the process holds your Dhan token, so do not expose it to a network.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from ..paper.config import ExperimentConfig
from ..paper.manager import ExperimentManager

STATIC: Final = Path(__file__).parent / "static"


def create_app(manager: ExperimentManager) -> FastAPI:
    app = FastAPI(title="AetherEngine Paper Trading Lab", docs_url="/api/docs", redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    def meta() -> dict:
        return manager.meta()

    @app.get("/api/experiments")
    def list_experiments() -> list[dict]:
        return manager.list()

    @app.post("/api/experiments", status_code=201)
    async def start_experiment(cfg: ExperimentConfig) -> dict:
        try:
            exp_id = await manager.start(cfg)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": exp_id}

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str) -> dict:
        detail = manager.get(exp_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return detail

    @app.post("/api/experiments/{exp_id}/stop")
    async def stop_experiment(exp_id: str) -> dict:
        if not await manager.stop(exp_id):
            raise HTTPException(status_code=404, detail="No running experiment with that id")
        return {"stopped": exp_id}

    @app.delete("/api/experiments/{exp_id}")
    def delete_experiment(exp_id: str) -> dict:
        try:
            deleted = manager.delete(exp_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {"deleted": exp_id}

    @app.get("/api/experiments/{exp_id}/trades.csv")
    def trades_csv(exp_id: str) -> Response:
        data = manager.trades_csv(exp_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return Response(
            data,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="trades-{exp_id}.csv"'},
        )

    return app


class PortalServer(uvicorn.Server):
    """uvicorn server that leaves SIGINT/SIGTERM to the engine's own shutdown."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def bind(host: str, port: int) -> socket.socket:
    """Bind up front so a busy port is a clear config error, not a uvicorn sys.exit."""
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        raise ValueError(f"Cannot open the portal on {host}:{port}: {exc.strerror}") from exc
    sock.set_inheritable(True)
    return sock


def make_server(app: FastAPI) -> PortalServer:
    return PortalServer(uvicorn.Config(app, log_level="warning", lifespan="off", access_log=False))
