#!/usr/bin/env python3
"""FastAPI status endpoint for Bird-ID service."""
from __future__ import annotations

import json
import os
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from bird_database import BirdDatabase
except ImportError:  # pragma: no cover - optional dependency
    BirdDatabase = None  # type: ignore

START_TIME = datetime.now(timezone.utc)
DATA_DIR = Path(os.environ.get("BIRDID_DATA_DIR", "/data"))
DB_PATH = Path(os.environ.get("BIRDID_DB_PATH", DATA_DIR / "birds.db"))
CONFIG_PATH = Path("/config/config.yaml")
RUNTIME_CONFIG_PATH = DATA_DIR / "runtime_config.json"
CLASSIFY_ENABLED = os.environ.get("BIRDID_CLASSIFY", "1") != "0"
EDGETPU_ENABLED = os.environ.get("BIRDID_USE_EDGETPU", "0") != "0"

# Load clips directory from config
def _load_clips_dir() -> Path:
    """Load clips directory from config, fallback to clips_360p."""
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
            clips_dir = config.get("video", {}).get("clips_dir", "/data/clips_360p")
            return Path(clips_dir)
    except Exception:
        return DATA_DIR / "clips_360p"

CLIPS_DIR = _load_clips_dir()

app = FastAPI(title="Bird-ID Service API")


class ConfigUpdate(BaseModel):
    """Runtime configuration update model."""
    source_mode: Optional[str] = None  # "live" or "video"
    video_file: Optional[str] = None
    use_edgetpu: Optional[bool] = None


def _query_recent(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent sightings from SQLite if available."""
    if not BirdDatabase or not DB_PATH.exists():
        return []
    try:
        with BirdDatabase(DB_PATH) as db:
            rows = db.get_recent_sightings(limit)
            return [
                {
                    "timestamp": row.get("timestamp"),
                    "common_name": row.get("common_name") or row.get("species"),
                    "confidence": row.get("confidence"),
                    "stream_source": row.get("stream_source"),
                }
                for row in rows
            ]
    except Exception:
        return []


@app.get("/api/healthz")
def healthcheck():
    """Simple health endpoint for Traefik / monitoring."""
    return {"status": "ok", "service": "birdid"}


@app.get("/api/status")
def status():
    """Service status (uptime, classifier mode, DB availability)."""
    uptime = (datetime.now(timezone.utc) - START_TIME).total_seconds()
    recent = _query_recent(limit=1)
    last_seen = recent[0]["timestamp"] if recent else None
    return {
        "service": "birdid",
        "uptime_seconds": int(uptime),
        "classify_enabled": CLASSIFY_ENABLED,
        "use_edgetpu": EDGETPU_ENABLED,
        "database_path": str(DB_PATH),
        "database_available": DB_PATH.exists(),
        "last_sighting": last_seen,
    }


@app.get("/api/recent")
def recent(limit: int = 10):
    """Return the most recent N sightings."""
    limit = max(1, min(limit, 50))
    return {"items": _query_recent(limit)}


@app.get("/api/species")
def species(limit: int = 10):
    """Return top species stats."""
    if not BirdDatabase or not DB_PATH.exists():
        return {"items": []}
    try:
        with BirdDatabase(DB_PATH) as db:
            rows = db.get_species_stats()[:limit]
        return {"items": rows}
    except Exception:
        return {"items": []}


@app.get("/api/videos")
def list_videos():
    """List all available test videos."""
    if not CLIPS_DIR.exists():
        return {"videos": []}

    videos = []
    for video_path in sorted(CLIPS_DIR.rglob("*.mp4")):
        rel_path = video_path.relative_to(CLIPS_DIR)
        category = rel_path.parent.name if rel_path.parent != Path(".") else "uncategorized"
        videos.append({
            "path": str(rel_path),
            "name": video_path.name,
            "category": category,
            "size_mb": round(video_path.stat().st_size / 1024 / 1024, 2)
        })

    return {"videos": videos, "total": len(videos)}


@app.get("/api/runtime-config")
def get_runtime_config():
    """Get current runtime configuration."""
    if not RUNTIME_CONFIG_PATH.exists():
        default = {
            "source_mode": "live",
            "video_file": None,
            "use_edgetpu": EDGETPU_ENABLED,
            "last_updated": None
        }
        return default

    try:
        with open(RUNTIME_CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")


@app.post("/api/runtime-config")
def update_runtime_config(update: ConfigUpdate):
    """Update runtime configuration (requires container restart to take effect)."""
    # Load current config
    if RUNTIME_CONFIG_PATH.exists():
        with open(RUNTIME_CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {
            "source_mode": "live",
            "video_file": None,
            "use_edgetpu": EDGETPU_ENABLED
        }

    # Update fields
    if update.source_mode is not None:
        if update.source_mode not in ("live", "video"):
            raise HTTPException(status_code=400, detail="source_mode must be 'live' or 'video'")
        config["source_mode"] = update.source_mode

    if update.video_file is not None:
        # Validate video exists
        video_path = CLIPS_DIR / update.video_file
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {update.video_file}")
        config["video_file"] = update.video_file

    if update.use_edgetpu is not None:
        config["use_edgetpu"] = update.use_edgetpu

    config["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Save config
    try:
        with open(RUNTIME_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    return {
        "success": True,
        "config": config,
        "message": "Configuration updated. Restart the container for changes to take effect: docker-compose restart birdid"
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("status_api:app", host="0.0.0.0", port=8000, reload=False)
