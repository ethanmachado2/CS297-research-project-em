# Archived experiment variants

This folder preserves notebooks, results and supporting code for the earlier resizing/crop, full-resolution, stability and single-image comparison experiments. Original project-relative paths are retained beneath this folder. `archive_manifest.json` records every archived file and its SHA-256 hash.

Git includes the archived notebooks, code, documentation, and manifest, while
`archive/outputs/` remains local and is ignored. The manifest describes the
complete local archive, so its result-file entries are not all present in a
fresh clone. Archived notebook outputs are retained as historical records;
active notebooks under `notebooks/` are distributed with their outputs cleared.

The original mixed pipeline/adapter test file is preserved here; its active counterpart retains pipeline tests and omits the archived adapter and severity-digit tests. Shared pipeline/evaluation modules remain intact in the active project to preserve existing behavior.

Archived notebooks and scripts may contain their original paths. Restore a variant and its dependencies to the original project locations before running it; do not overwrite active files without checking them. Saved records retain original paths as historical evidence.

The active `outputs/recovered_history/` folder is retained as provenance for the before-resize-cap configuration. Active new results remain under `outputs/original_configuration_runs/` and `outputs/original_configuration_few_shot_runs/`.
