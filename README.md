# LeRobot Annotate

A lightweight web UI for manually editing LeRobot v3.1 language annotations. It works with local or Hugging Face Hub datasets and exports parquet files aligned with the `language_persistent` / `language_events` schema used by the LeRobot language-column and annotation-pipeline PRs.

## What it produces

- Updated `data/chunk-*/file-*.parquet` files with:
  - `language_persistent`: broadcast episode-level atoms for `subtask`, `plan`, and `memory`
  - `language_events`: exact-frame atoms for `interjection`, `vqa`, and speech tool calls
  - `tools`: the dataset-level `say` tool schema used by speech atoms
- Legacy `subtask_index`, `meta/subtasks.parquet`, and `meta/tasks_high_level.parquet` are removed on export.
- Annotation state is also saved to `meta/lerobot_annotations.json` for fast resume.

Each language atom has the shared row shape:

```json
{
  "role": "user | assistant | system | tool",
  "content": "text payload or null",
  "style": "subtask | plan | memory | interjection | vqa | null",
  "timestamp": 0.0,
  "tool_calls": null
}
```

Speech is represented as a style-less assistant atom with a `say` tool call. VQA assistant answers are JSON strings and can store bounding boxes, keypoints, counts, attributes, or spatial relations.

## Local usage

```bash
cd /admin/home/jade_choghari/lerobot-annotate
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 7860
```

Open `http://localhost:7860` in your browser.

### Workflow

1. **Connect dataset**
   - For HF datasets: enter repo ID (e.g. `lerobot/your-dataset`).
   - For local datasets: enter the dataset root path.
2. **Select an episode** from the left list to open its video.
3. **Edit persistent language atoms** for subtasks, plans, and memory.
4. **Edit exact-frame event atoms** for user interjections, robot speech, and VQA.
5. **Draw VQA boxes and points** directly on the video; boxes and points are rendered back on playback.
6. **Save episode** to persist annotations.
7. **Export** to write parquet updates and dataset metadata.

## Hugging Face Spaces (Docker)

1. Create a new Space and select **Docker**.
2. Point it to this repository.
3. The Space will build the provided `Dockerfile` and run `uvicorn` on port `7860`.

Optional environment variables:

- `LEROBOT_ANNOTATE_CACHE`: where HF datasets are downloaded (default `/tmp/lerobot_annotate_cache`).
- `LEROBOT_ANNOTATE_EXPORT`: where exports are written (default `/tmp/lerobot_annotate_exports`).

## Notes

- The tool only downloads `meta/` for Hub datasets on load. Video and data parquet files are fetched on demand when you open an episode or inspect frame timestamps.
- If an existing dataset already has `language_persistent` and `language_events`, the editor reads those columns as the initial annotation state when no `meta/lerobot_annotations.json` file is present.
- Event atoms are snapped to source frame timestamps so render-time exact matching works.
- Exports copy `meta/` and `data/` into a new output directory. Videos are symlinked by default (toggle “Copy videos” to duplicate them).
