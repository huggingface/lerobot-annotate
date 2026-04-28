"""LeRobot Annotate backend.

Aligned with the LeRobot v3.1 language schema introduced in PR1 (lerobot#3467)
and the steerable annotation pipeline conventions from PR2 (lerobot#3471):

- ``data/chunk-*/file-*.parquet`` is rewritten with two new columns:
    * ``language_persistent: list<struct{role,content,style,timestamp,tool_calls}>``
      — broadcast identically across every frame in the episode. Holds
      ``subtask``, ``plan``, ``memory``.
    * ``language_events: list<struct{role,content,style,timestamp,tool_calls}>``
      — only populated on the exact frames where events were emitted. Holds
      ``interjection``, ``vqa``, and speech tool-call atoms (``style=None``).
- The legacy ``subtask_index`` column is dropped.
- A dataset-level ``tools`` column carries the JSON schema for the ``say`` tool.

The UI is structured around five styles:
- subtask / plan / memory  → persistent
- interjection (+ paired speech tool call) → events
- vqa (with bbox / point / count / attribute / spatial answers) → events

VQA bounding boxes and points are JSON-encoded into the ``content`` of the
assistant turn, matching the schema used by the steerable annotation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
CACHE_ROOT = Path(os.environ.get("LEROBOT_ANNOTATE_CACHE", "/tmp/lerobot_annotate_cache"))
EXPORT_ROOT = Path(os.environ.get("LEROBOT_ANNOTATE_EXPORT", "/tmp/lerobot_annotate_exports"))
TRIMMED_VIDEO_CACHE = CACHE_ROOT / "trimmed_videos"

# --- Schema constants (mirror src/lerobot/datasets/language.py) -----------------

PERSISTENT_STYLES = {"subtask", "plan", "memory"}
EVENT_ONLY_STYLES = {"interjection", "vqa"}
KNOWN_STYLES = PERSISTENT_STYLES | EVENT_ONLY_STYLES
LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"

SAY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The verbatim text to speak."},
            },
            "required": ["text"],
        },
    },
}


def column_for_style(style: str | None) -> str:
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")


# --- FFmpeg helpers (unchanged from the original tool) --------------------------


def trim_video_with_ffmpeg(input_path: Path, output_path: Path, start_time: float, end_time: float) -> bool:
    duration = end_time - start_time
    if duration <= 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            cmd_reencode = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", str(input_path),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                str(output_path),
            ]
            result = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"FFmpeg re-encode error: {result.stderr}")
                return False
        return output_path.exists()
    except subprocess.TimeoutExpired:
        print("FFmpeg timed out")
        return False
    except FileNotFoundError:
        print("FFmpeg not found - please install FFmpeg")
        return False
    except Exception as e:
        print(f"Error trimming video: {e}")
        return False


def get_trimmed_video_cache_path(video_path: Path, episode_index: int, start_time: float, end_time: float) -> Path:
    key = f"{video_path}_{episode_index}_{start_time:.3f}_{end_time:.3f}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
    return TRIMMED_VIDEO_CACHE / f"ep{episode_index}_{hash_key}.mp4"


def get_video_duration(video_path: Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return 0.0


# --- Pydantic payloads ---------------------------------------------------------


class DatasetLoadRequest(BaseModel):
    source: str  # "hf" or "local"
    repo_id: str | None = None
    revision: str | None = None
    local_path: str | None = None
    video_key: str | None = None


class LanguageAtom(BaseModel):
    """One row from `language_persistent` or `language_events`."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | None = None
    style: str | None = None  # subtask|plan|memory|interjection|vqa or None for speech
    timestamp: float
    tool_calls: list[dict[str, Any]] | None = None


class EpisodeAnnotationsPayload(BaseModel):
    episode_index: int
    atoms: list[LanguageAtom] = []


@dataclass
class EpisodeAnnotations:
    atoms: list[dict[str, Any]] = field(default_factory=list)


# --- Legacy migration ----------------------------------------------------------


def _legacy_to_atoms(legacy: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert v1 (subtasks + high_levels) episode annotations into atoms.

    - subtasks ``{start,end,label}`` → persistent ``subtask`` row at ``start``.
    - high_levels ``{start,user_prompt,robot_utterance,...}`` →
        - user ``interjection`` event at ``start`` with content=user_prompt
        - assistant speech tool-call atom (style=None) at ``start`` with
          ``tool_calls=[{type:function, function:{name:'say',
          arguments:{text:robot_utterance}}}]``.
    """
    atoms: list[dict[str, Any]] = []
    for seg in legacy.get("subtasks", []) or []:
        if "label" not in seg or "start" not in seg:
            continue
        atoms.append(
            {
                "role": "assistant",
                "content": str(seg["label"]),
                "style": "subtask",
                "timestamp": float(seg["start"]),
                "tool_calls": None,
            }
        )
    for seg in legacy.get("high_levels", []) or []:
        ts = float(seg.get("start", 0.0))
        prompt = (seg.get("user_prompt") or "").strip()
        utter = (seg.get("robot_utterance") or "").strip()
        if prompt:
            atoms.append(
                {
                    "role": "user",
                    "content": prompt,
                    "style": "interjection",
                    "timestamp": ts,
                    "tool_calls": None,
                }
            )
        if utter:
            atoms.append(
                {
                    "role": "assistant",
                    "content": None,
                    "style": None,
                    "timestamp": ts,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "say", "arguments": {"text": utter}},
                        }
                    ],
                }
            )
    return atoms


# --- DataManager ---------------------------------------------------------------


class DataManager:
    def __init__(self) -> None:
        self.source: str | None = None
        self.repo_id: str | None = None
        self.revision: str | None = None
        self.dataset_root: Path | None = None
        self.info: dict[str, Any] | None = None
        self.episodes_df: pd.DataFrame | None = None
        self.video_key: str | None = None
        self.annotations: dict[int, EpisodeAnnotations] = {}
        self.annotations_path: Path | None = None
        # Cached frame timestamps per episode, loaded lazily from data parquet.
        self._frame_ts_cache: dict[int, list[float]] = {}

    def load_dataset(self, req: DatasetLoadRequest) -> dict[str, Any]:
        if req.source not in {"hf", "local"}:
            raise HTTPException(status_code=400, detail="source must be 'hf' or 'local'")

        self.source = req.source
        self.repo_id = req.repo_id
        self.revision = req.revision
        self._frame_ts_cache = {}

        if req.source == "local":
            if not req.local_path:
                raise HTTPException(status_code=400, detail="local_path is required for local source")
            root = Path(req.local_path).expanduser().resolve()
            if not root.exists():
                raise HTTPException(status_code=404, detail=f"Dataset path not found: {root}")
            self.dataset_root = root
        else:
            if not req.repo_id:
                raise HTTPException(status_code=400, detail="repo_id is required for hf source")
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            repo_dir = CACHE_ROOT / req.repo_id.replace("/", "__")
            repo_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                req.repo_id,
                repo_type="dataset",
                revision=req.revision,
                local_dir=repo_dir,
                allow_patterns=["meta/*"],
            )
            self.dataset_root = repo_dir

        self.info = self._load_info(self.dataset_root)
        self.episodes_df = self._load_episodes(self.dataset_root)

        video_keys = self._get_video_keys()
        if not video_keys:
            raise HTTPException(status_code=400, detail="Dataset has no video keys")
        self.video_key = req.video_key or video_keys[0]
        if self.video_key not in video_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Video key '{self.video_key}' not found. Available: {', '.join(video_keys)}",
            )

        self.annotations_path = self.dataset_root / "meta" / "lerobot_annotations.json"
        self._load_existing_annotations()
        return self._build_summary()

    def _load_info(self, root: Path) -> dict[str, Any]:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing info.json at {info_path}")
        return json.loads(info_path.read_text())

    def _load_episodes(self, root: Path) -> pd.DataFrame:
        episodes_root = root / "meta" / "episodes"
        if not episodes_root.exists():
            raise HTTPException(status_code=404, detail=f"Missing episodes directory at {episodes_root}")
        files = sorted(episodes_root.rglob("*.parquet"))
        if not files:
            raise HTTPException(status_code=404, detail="No episodes parquet files found")
        dfs = [pd.read_parquet(path) for path in files]
        df = pd.concat(dfs, ignore_index=True)
        if "episode_index" not in df.columns:
            raise HTTPException(status_code=400, detail="episodes parquet missing 'episode_index' column")
        return df.sort_values("episode_index").reset_index(drop=True)

    def _get_video_keys(self) -> list[str]:
        features = self.info.get("features", {}) if self.info else {}
        return sorted([key for key, meta in features.items() if meta.get("dtype") == "video"])

    def _load_existing_annotations(self) -> None:
        self.annotations = {}
        if self.annotations_path and self.annotations_path.exists():
            data = json.loads(self.annotations_path.read_text())
            version = int(data.get("version", 1))
            episodes = data.get("episodes", {})
            for ep_str, payload in episodes.items():
                ep_idx = int(ep_str)
                if version >= 2 and "atoms" in payload:
                    self.annotations[ep_idx] = EpisodeAnnotations(
                        atoms=[dict(a) for a in payload.get("atoms", [])]
                    )
                else:
                    # v1 legacy: subtasks + high_levels → atoms
                    self.annotations[ep_idx] = EpisodeAnnotations(atoms=_legacy_to_atoms(payload))
            return

        # Even older fallback: meta/skills.json
        skills_path = self.dataset_root / "meta" / "skills.json"
        if skills_path.exists():
            data = json.loads(skills_path.read_text())
            for ep_str, payload in data.get("episodes", {}).items():
                ep_idx = int(ep_str)
                skills = payload.get("skills", [])
                legacy = {
                    "subtasks": [
                        {"start": s["start"], "end": s["end"], "label": s["name"]}
                        for s in skills
                        if "start" in s and "end" in s
                    ],
                    "high_levels": [],
                }
                self.annotations[ep_idx] = EpisodeAnnotations(atoms=_legacy_to_atoms(legacy))

    def _save_annotations(self) -> None:
        if not self.annotations_path:
            return
        payload = {
            "version": 2,
            "schema": {
                "persistent_styles": sorted(PERSISTENT_STYLES),
                "event_styles": sorted(EVENT_ONLY_STYLES),
            },
            "episodes": {
                str(ep_idx): {"atoms": ann.atoms} for ep_idx, ann in self.annotations.items()
            },
        }
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path.write_text(json.dumps(payload, indent=2))

    def _build_summary(self) -> dict[str, Any]:
        assert self.info and self.episodes_df is not None
        fps = float(self.info.get("fps", 30))
        video_keys = self._get_video_keys()
        video_key = self.video_key or (video_keys[0] if video_keys else None)
        episode_video_offsets = self._calculate_video_offsets(video_key, fps) if video_key else {}

        episodes = []
        for _, row in self.episodes_df.iterrows():
            length = int(row.get("length", row.get("dataset_to_index", 0) - row.get("dataset_from_index", 0)))
            duration = length / fps if fps else 0.0
            ep_idx = int(row["episode_index"])
            video_info = episode_video_offsets.get(ep_idx, {"video_start_time": 0.0, "video_end_time": duration})
            episodes.append(
                {
                    "episode_index": ep_idx,
                    "length": length,
                    "duration": duration,
                    "video_start_time": video_info["video_start_time"],
                    "video_end_time": video_info["video_end_time"],
                }
            )
        return {
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "root": str(self.dataset_root),
            "fps": fps,
            "video_keys": video_keys,
            "selected_video_key": self.video_key,
            "persistent_styles": sorted(PERSISTENT_STYLES),
            "event_styles": sorted(EVENT_ONLY_STYLES),
            "episodes": episodes,
        }

    def _calculate_video_offsets(self, video_key: str, fps: float) -> dict[int, dict[str, float]]:
        if self.episodes_df is None:
            return {}
        from_ts_col = f"videos/{video_key}/from_timestamp"
        to_ts_col = f"videos/{video_key}/to_timestamp"
        has_timestamp_cols = from_ts_col in self.episodes_df.columns and to_ts_col in self.episodes_df.columns
        result = {}
        for _, row in self.episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            length = int(row.get("length", row.get("dataset_to_index", 0) - row.get("dataset_from_index", 0)))
            duration = length / fps if fps else 0.0
            if has_timestamp_cols:
                from_ts = row.get(from_ts_col)
                to_ts = row.get(to_ts_col)
                if pd.notna(from_ts) and pd.notna(to_ts):
                    result[ep_idx] = {"video_start_time": float(from_ts), "video_end_time": float(to_ts)}
                else:
                    result[ep_idx] = {"video_start_time": 0.0, "video_end_time": duration}
            else:
                result[ep_idx] = {"video_start_time": 0.0, "video_end_time": duration}
        return result

    def get_episode_video_path(self, episode_index: int, video_key: str | None = None) -> Path:
        if self.episodes_df is None or self.info is None:
            raise HTTPException(status_code=400, detail="Dataset not loaded")
        video_key = video_key or self.video_key
        if not video_key:
            raise HTTPException(status_code=400, detail="video_key is required")

        row = self.episodes_df[self.episodes_df["episode_index"] == episode_index]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Episode {episode_index} not found")
        row = row.iloc[0]

        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        if chunk_col not in row or file_col not in row:
            raise HTTPException(status_code=400, detail=f"Video key '{video_key}' not available for this dataset")

        chunk_index = int(row[chunk_col])
        file_index = int(row[file_col])
        rel_path = self.info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        rel_path = rel_path.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
        full_path = (self.dataset_root / rel_path).resolve()

        if full_path.exists():
            return full_path

        if self.source == "hf" and self.repo_id:
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=rel_path,
                revision=self.revision,
                local_dir=self.dataset_root,
            )
            if full_path.exists():
                return full_path

        raise HTTPException(status_code=404, detail=f"Video file not found: {full_path}")

    def get_episode_data_path(self, episode_index: int) -> Path | None:
        """Find the data parquet file containing this episode (for frame timestamps)."""
        if self.episodes_df is None or self.info is None or self.dataset_root is None:
            return None
        row = self.episodes_df[self.episodes_df["episode_index"] == episode_index]
        if row.empty:
            return None
        row = row.iloc[0]
        chunk_col = "data/chunk_index"
        file_col = "data/file_index"
        if chunk_col not in row or file_col not in row:
            return None
        chunk_index = int(row[chunk_col])
        file_index = int(row[file_col])
        rel = self.info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        rel = rel.format(chunk_index=chunk_index, file_index=file_index)
        full = (self.dataset_root / rel).resolve()
        if full.exists():
            return full
        if self.source == "hf" and self.repo_id:
            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=rel,
                    revision=self.revision,
                    local_dir=self.dataset_root,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[frame_ts] download failed for {rel}: {e}")
                return None
        return full if full.exists() else None

    def get_frame_timestamps(self, episode_index: int) -> list[float]:
        if episode_index in self._frame_ts_cache:
            return self._frame_ts_cache[episode_index]
        path = self.get_episode_data_path(episode_index)
        if path is None:
            return []
        try:
            df = pd.read_parquet(path, columns=["episode_index", "timestamp"])
        except Exception as e:  # noqa: BLE001
            print(f"[frame_ts] read failed for ep {episode_index}: {e}")
            return []
        ts = df.loc[df["episode_index"] == episode_index, "timestamp"].astype(float).tolist()
        ts.sort()
        self._frame_ts_cache[episode_index] = ts
        return ts

    def snap_to_frame(self, episode_index: int, ts: float) -> float:
        """Snap an arbitrary timestamp to the closest frame in the episode."""
        frame_ts = self.get_frame_timestamps(episode_index)
        if not frame_ts:
            return float(ts)
        # Linear scan is fine — episodes are typically a few thousand frames.
        return min(frame_ts, key=lambda f: abs(f - ts))

    def get_episode_annotations(self, episode_index: int) -> EpisodeAnnotations:
        if episode_index not in self.annotations:
            self.annotations[episode_index] = EpisodeAnnotations(
                atoms=self._load_episode_atoms_from_data(episode_index)
            )
        return self.annotations[episode_index]

    def set_episode_annotations(self, payload: EpisodeAnnotationsPayload) -> None:
        atoms = [a.dict() for a in payload.atoms]
        for atom in atoms:
            self._validate_atom(atom)
        self.annotations[payload.episode_index] = EpisodeAnnotations(atoms=atoms)
        self._save_annotations()

    def _validate_atom(self, atom: dict[str, Any]) -> None:
        style = atom.get("style")
        if style is not None and style not in KNOWN_STYLES:
            raise HTTPException(status_code=400, detail=f"Unknown language style: {style!r}")
        has_content = atom.get("content") is not None
        has_tools = bool(atom.get("tool_calls"))
        if not (has_content or has_tools):
            raise HTTPException(status_code=400, detail="atom must have content or tool_calls")
        if style is None and not has_tools:
            raise HTTPException(status_code=400, detail="style=None requires tool_calls (speech atom)")

    def _load_episode_atoms_from_data(self, episode_index: int) -> list[dict[str, Any]]:
        """Read existing v3.1 language columns for one episode, if present.

        This lets the Space edit datasets that were already annotated by PR2's
        pipeline, even when there is no ``meta/lerobot_annotations.json`` side
        state. The JSON side state remains the fast resume path once a user
        saves edits in the UI.
        """
        path = self.get_episode_data_path(episode_index)
        if path is None:
            return []
        try:
            schema = pq.read_schema(path)
            available = set(schema.names)
            if LANGUAGE_PERSISTENT not in available and LANGUAGE_EVENTS not in available:
                return []
            columns = ["episode_index"]
            if LANGUAGE_PERSISTENT in available:
                columns.append(LANGUAGE_PERSISTENT)
            if LANGUAGE_EVENTS in available:
                columns.append(LANGUAGE_EVENTS)
            table = pq.read_table(path, columns=columns)
        except Exception as e:  # noqa: BLE001
            print(f"[annotations] could not read language columns for ep {episode_index}: {e}")
            return []
        return self._extract_atoms_from_table(table, episode_index)

    def _extract_atoms_from_table(self, table: pa.Table, episode_index: int) -> list[dict[str, Any]]:
        if "episode_index" not in table.column_names:
            return []

        episode_col = table.column("episode_index").to_pylist()
        persistent_col = (
            table.column(LANGUAGE_PERSISTENT).to_pylist()
            if LANGUAGE_PERSISTENT in table.column_names
            else None
        )
        events_col = (
            table.column(LANGUAGE_EVENTS).to_pylist()
            if LANGUAGE_EVENTS in table.column_names
            else None
        )

        atoms: list[dict[str, Any]] = []
        seen: set[str] = set()
        persistent_loaded = False

        def add_many(raw_atoms: Any) -> None:
            if not raw_atoms:
                return
            for raw in raw_atoms:
                atom = self._coerce_existing_atom(raw)
                if atom is None:
                    continue
                key = json.dumps(atom, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                atoms.append(atom)

        for row_idx, ep_value in enumerate(episode_col):
            if int(ep_value) != int(episode_index):
                continue
            # Persistent rows are broadcast identically on every frame; read
            # them once from the first matching row.
            if persistent_col is not None and not persistent_loaded:
                add_many(persistent_col[row_idx])
                persistent_loaded = True
            if events_col is not None:
                add_many(events_col[row_idx])

        atoms.sort(key=lambda a: (a["timestamp"], a.get("style") or "", a.get("role") or ""))
        return atoms

    @staticmethod
    def _coerce_existing_atom(raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            try:
                raw = dict(raw)
            except Exception:  # noqa: BLE001
                return None
        if not raw.get("role"):
            return None
        tool_calls = raw.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        return {
            "role": str(raw["role"]),
            "content": None if raw.get("content") is None else str(raw.get("content")),
            "style": raw.get("style"),
            "timestamp": float(raw.get("timestamp", 0.0)),
            "tool_calls": tool_calls or None,
        }

    # --- Export -----------------------------------------------------------

    def export_dataset(self, output_dir: str | None = None, copy_videos: bool = False) -> dict[str, Any]:
        if self.dataset_root is None or self.info is None:
            raise HTTPException(status_code=400, detail="Dataset not loaded")

        if output_dir:
            out_root = Path(output_dir).expanduser().resolve()
        else:
            EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
            name = (self.repo_id or "local_dataset").replace("/", "__")
            out_root = EXPORT_ROOT / f"{name}_annotated"

        out_root.mkdir(parents=True, exist_ok=True)

        # Copy meta directory
        src_meta = self.dataset_root / "meta"
        dst_meta = out_root / "meta"
        if dst_meta.exists():
            shutil.rmtree(dst_meta)
        shutil.copytree(src_meta, dst_meta)

        # Update info.json features:
        #  - drop subtask_index
        #  - add language_persistent / language_events / tools
        info_path = dst_meta / "info.json"
        info = json.loads(info_path.read_text())
        info.setdefault("features", {})
        info["features"].pop("subtask_index", None)
        info["features"][LANGUAGE_PERSISTENT] = {"dtype": "language", "shape": [1], "names": None}
        info["features"][LANGUAGE_EVENTS] = {"dtype": "language", "shape": [1], "names": None}
        info["features"]["tools"] = {"dtype": "string", "shape": [1], "names": None}
        info_path.write_text(json.dumps(info, indent=2))

        # Drop legacy meta files
        for legacy in ("subtasks.parquet", "tasks_high_level.parquet"):
            p = dst_meta / legacy
            if p.exists():
                p.unlink()

        # Update data files
        data_dir = self.dataset_root / "data"
        data_files = sorted(data_dir.rglob("*.parquet"))
        if not data_files:
            # Hub datasets may have no data downloaded yet — try downloading.
            if self.source == "hf" and self.repo_id:
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    local_dir=self.dataset_root,
                    allow_patterns=["data/**/*.parquet"],
                )
                data_files = sorted(data_dir.rglob("*.parquet"))
        if not data_files:
            raise HTTPException(status_code=404, detail="No data parquet files found")

        n_persistent = 0
        n_events = 0
        for src_path in data_files:
            rel_path = src_path.relative_to(self.dataset_root)
            dst_path = out_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(src_path)
            new_table, ep_persistent, ep_events = self._materialize_table(table)
            n_persistent += ep_persistent
            n_events += ep_events
            pq.write_table(new_table, dst_path)

        # Copy or link videos
        src_videos = self.dataset_root / "videos"
        dst_videos = out_root / "videos"
        if src_videos.exists():
            if dst_videos.exists():
                shutil.rmtree(dst_videos)
            if copy_videos:
                shutil.copytree(src_videos, dst_videos)
            else:
                try:
                    os.symlink(src_videos, dst_videos)
                except OSError:
                    shutil.copytree(src_videos, dst_videos)

        return {
            "output_dir": str(out_root),
            "persistent_rows": n_persistent,
            "event_rows": n_events,
        }

    def _materialize_table(self, table: pa.Table) -> tuple[pa.Table, int, int]:
        """Add language_persistent / language_events / tools to a data parquet shard."""
        if "episode_index" not in table.column_names or "timestamp" not in table.column_names:
            raise HTTPException(
                status_code=400,
                detail="data parquet missing 'episode_index' or 'timestamp' columns",
            )

        episode_col = table.column("episode_index").to_pylist()
        ts_col = [float(x) for x in table.column("timestamp").to_pylist()]
        n_rows = table.num_rows

        # Pre-bucket atoms by episode
        persistent_by_ep: dict[int, list[dict[str, Any]]] = {}
        events_by_ep_ts: dict[int, dict[float, list[dict[str, Any]]]] = {}

        n_persistent_total = 0
        n_event_total = 0

        unique_eps = sorted(set(episode_col))
        for ep_idx in unique_eps:
            ann = self.annotations.get(int(ep_idx))
            ann_atoms = ann.atoms if ann and ann.atoms else self._extract_atoms_from_table(table, int(ep_idx))
            if not ann_atoms:
                persistent_by_ep[ep_idx] = []
                events_by_ep_ts[ep_idx] = {}
                continue

            persistent_rows: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []
            # Build per-episode frame-timestamp set for snapping events
            frame_ts = [ts_col[i] for i in range(n_rows) if episode_col[i] == ep_idx]
            frame_ts.sort()

            for atom in ann_atoms:
                style = atom.get("style")
                normalized = self._normalize_atom(atom)
                col = column_for_style(style)
                if col == LANGUAGE_PERSISTENT:
                    persistent_rows.append(normalized)
                else:
                    if frame_ts:
                        snapped = min(frame_ts, key=lambda f: abs(f - normalized["timestamp"]))
                        normalized["timestamp"] = float(snapped)
                    event_rows.append(normalized)

            persistent_rows.sort(key=lambda r: (r["timestamp"], r.get("style") or "", r.get("role") or ""))
            persistent_by_ep[ep_idx] = persistent_rows

            buckets: dict[float, list[dict[str, Any]]] = {}
            for r in event_rows:
                buckets.setdefault(r["timestamp"], []).append(r)
            for ts in buckets:
                buckets[ts].sort(key=lambda r: (r.get("style") or "", r.get("role") or ""))
            events_by_ep_ts[ep_idx] = buckets

            n_persistent_total += len(persistent_rows)
            n_event_total += sum(len(v) for v in buckets.values())

        per_row_persistent = [persistent_by_ep.get(episode_col[i], []) for i in range(n_rows)]
        per_row_events = [
            events_by_ep_ts.get(episode_col[i], {}).get(ts_col[i], []) for i in range(n_rows)
        ]

        # Build the new table column-by-column
        keep_names = []
        keep_cols = []
        for name in table.column_names:
            if name == "subtask_index":
                continue  # drop legacy
            if name in {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS, "tools"}:
                continue  # rebuild
            keep_names.append(name)
            keep_cols.append(table.column(name))

        persistent_arr = pa.array(per_row_persistent)
        events_arr = pa.array(per_row_events)
        tools_json = json.dumps([SAY_TOOL_SCHEMA], sort_keys=True)
        tools_arr = pa.array([tools_json] * n_rows, type=pa.string())

        new_names = keep_names + [LANGUAGE_PERSISTENT, LANGUAGE_EVENTS, "tools"]
        new_cols = keep_cols + [persistent_arr, events_arr, tools_arr]
        return pa.Table.from_arrays(new_cols, names=new_names), n_persistent_total, n_event_total

    @staticmethod
    def _normalize_atom(atom: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": str(atom["role"]),
            "content": None if atom.get("content") is None else str(atom["content"]),
            "style": atom.get("style"),
            "timestamp": float(atom.get("timestamp", 0.0)),
            "tool_calls": list(atom["tool_calls"]) if atom.get("tool_calls") else None,
        }


def parse_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1
    if start >= file_size:
        return None
    end = min(end, file_size - 1)
    return start, end


# --- FastAPI app --------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = DataManager()


@app.get("/")
def root() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>LeRobot Annotate</h1><p>Missing static index.html</p>")
    return HTMLResponse(index_path.read_text())


@app.post("/api/dataset/load")
def load_dataset(req: DatasetLoadRequest) -> JSONResponse:
    summary = manager.load_dataset(req)
    return JSONResponse(summary)


@app.get("/api/dataset/info")
def dataset_info() -> JSONResponse:
    if manager.info is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    return JSONResponse(manager._build_summary())


@app.get("/api/episodes/{episode_index}/annotations")
def get_annotations(episode_index: int) -> JSONResponse:
    ann = manager.get_episode_annotations(episode_index)
    return JSONResponse({"episode_index": episode_index, "atoms": ann.atoms})


@app.post("/api/episodes/{episode_index}/annotations")
def set_annotations(episode_index: int, payload: EpisodeAnnotationsPayload) -> JSONResponse:
    if episode_index != payload.episode_index:
        raise HTTPException(status_code=400, detail="Episode index mismatch")
    manager.set_episode_annotations(payload)
    return JSONResponse({"ok": True})


@app.get("/api/episodes/{episode_index}/frame_timestamps")
def get_episode_frame_timestamps(episode_index: int) -> JSONResponse:
    """Return the list of frame timestamps for this episode (in seconds).

    Used by the UI to snap event-style atoms (interjection, vqa, speech) to
    exact frame timestamps, matching the requirement enforced by the
    steerable-pipeline writer in lerobot#3471.
    """
    ts = manager.get_frame_timestamps(episode_index)
    return JSONResponse({"episode_index": episode_index, "timestamps": ts})


@app.post("/api/export")
def export_dataset(payload: dict[str, Any]) -> JSONResponse:
    output_dir = payload.get("output_dir")
    copy_videos = bool(payload.get("copy_videos", False))
    result = manager.export_dataset(output_dir=output_dir, copy_videos=copy_videos)
    return JSONResponse(result)


class PushToHubRequest(BaseModel):
    hf_token: str
    push_in_place: bool = True
    new_repo_id: str | None = None
    private: bool = False
    commit_message: str = "Add language annotations from LeRobot Annotate"


@app.post("/api/push_to_hub")
def push_to_hub(payload: PushToHubRequest) -> JSONResponse:
    if manager.dataset_root is None or manager.info is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    if manager.source != "hf":
        raise HTTPException(status_code=400, detail="Can only push to Hub for datasets loaded from Hub")

    data_dir = manager.dataset_root / "data"
    data_files_exist = data_dir.exists() and list(data_dir.rglob("*.parquet"))
    if not data_files_exist:
        if not manager.repo_id:
            raise HTTPException(status_code=400, detail="No repo ID available to download data files")
        snapshot_download(
            manager.repo_id,
            repo_type="dataset",
            revision=manager.revision,
            local_dir=manager.dataset_root,
            allow_patterns=["data/**/*.parquet"],
        )

    videos_dir = manager.dataset_root / "videos"
    if not videos_dir.exists() and manager.repo_id:
        try:
            snapshot_download(
                manager.repo_id,
                repo_type="dataset",
                revision=manager.revision,
                local_dir=manager.dataset_root,
                allow_patterns=["videos/**/*.mp4"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[Push to Hub] Warning: could not download videos: {e}")

    export_result = manager.export_dataset(copy_videos=True)
    export_dir = Path(export_result["output_dir"])

    if payload.push_in_place:
        if not manager.repo_id:
            raise HTTPException(status_code=400, detail="No original repo ID found")
        target_repo = manager.repo_id
    else:
        if not payload.new_repo_id:
            raise HTTPException(status_code=400, detail="New repo ID is required when not pushing in place")
        target_repo = payload.new_repo_id

    api = HfApi(token=payload.hf_token)
    if not payload.push_in_place:
        api.create_repo(
            repo_id=target_repo,
            repo_type="dataset",
            private=payload.private,
            exist_ok=True,
        )

    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=target_repo,
        repo_type="dataset",
        commit_message=payload.commit_message,
    )

    return JSONResponse(
        {
            "ok": True,
            "repo_id": target_repo,
            "url": f"https://huggingface.co/datasets/{target_repo}",
            "message": f"Successfully pushed to {target_repo}",
        }
    )


@app.get("/api/episodes/{episode_index}/video_timing")
def get_episode_video_timing(episode_index: int, video_key: str | None = None) -> JSONResponse:
    if manager.episodes_df is None or manager.info is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    video_key = video_key or manager.video_key
    fps = float(manager.info.get("fps", 30))
    row = manager.episodes_df[manager.episodes_df["episode_index"] == episode_index]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Episode {episode_index} not found")
    row = row.iloc[0]
    length = int(row.get("length", row.get("dataset_to_index", 0) - row.get("dataset_from_index", 0)))
    duration = length / fps if fps else 0.0
    video_offsets = manager._calculate_video_offsets(video_key, fps) if video_key else {}
    video_info = video_offsets.get(episode_index, {"video_start_time": 0.0, "video_end_time": duration})
    return JSONResponse(
        {
            "episode_index": episode_index,
            "fps": fps,
            "length": length,
            "duration": duration,
            "video_start_time": video_info["video_start_time"],
            "video_end_time": video_info["video_end_time"],
        }
    )


@app.get("/api/video/{episode_index}")
def stream_video(episode_index: int, request: Request, video_key: str | None = None) -> Response:
    video_key = video_key or manager.video_key
    original_path = manager.get_episode_video_path(episode_index, video_key=video_key)
    fps = float(manager.info.get("fps", 30)) if manager.info else 30.0
    video_offsets = manager._calculate_video_offsets(video_key, fps) if video_key else {}
    video_info = video_offsets.get(episode_index, {"video_start_time": 0.0, "video_end_time": 0.0})
    start_time = video_info["video_start_time"]
    end_time = video_info["video_end_time"]
    needs_trimming = start_time > 0.1 or (end_time > 0 and end_time < get_video_duration(original_path) - 0.5)

    if needs_trimming and end_time > start_time:
        cache_path = get_trimmed_video_cache_path(original_path, episode_index, start_time, end_time)
        if not cache_path.exists():
            print(f"Trimming video for episode {episode_index}: {start_time:.2f}s - {end_time:.2f}s")
            success = trim_video_with_ffmpeg(original_path, cache_path, start_time, end_time)
            path = cache_path if success else original_path
        else:
            path = cache_path
    else:
        path = original_path

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        byte_range = parse_range(range_header, file_size)
        if not byte_range:
            return Response(status_code=416)
        start, end = byte_range
        length = end - start + 1

        def iterfile() -> Any:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        return StreamingResponse(iterfile(), status_code=206, media_type="video/mp4", headers=headers)

    return FileResponse(path, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
