from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
LOGGER_NAME = "dataset_x_splitter"
DEFAULT_IMAGES_DIR = Path("ml_data") / "fsoco_raw" / "ampera" / "img"
DEFAULT_LABELS_DIR = Path("ml_data") / "fsoco_raw" / "ampera" / "ann"
DEFAULT_ASPECT_RATIO = 1
DEFAULT_TILE_OVERLAP_RATIO = 0.15
DEFAULT_NEGATIVE_TILE_KEEP_RATIO = 0.2
DEFAULT_ASSIGNMENT_MODE = "visible"
DEFAULT_MIN_VISIBLE_FRACTION = 1.0
DEFAULT_MAX_TASKS_PER_CHILD = 128
DEFAULT_IN_FLIGHT_MULTIPLIER = 4
VALID_ASSIGNMENT_MODES = {"center", "visible"}


@dataclass(frozen=True)
class Box:
    class_title: str
    class_id: int | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    source_object: dict[str, Any] | None = None

    def shifted(self, dx: float, dy: float = 0.0) -> "Box":
        return Box(
            class_title=self.class_title,
            class_id=self.class_id,
            x_min=self.x_min + dx,
            y_min=self.y_min + dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
            source_object=self.source_object,
        )


@dataclass
class SplitStats:
    target_ratio: float
    total_images: int = 0
    images_with_label_file: int = 0
    runs_skipped_existing_outputs: int = 0
    images_missing_label_file: int = 0
    images_deleted_missing_label_file: int = 0
    images_failed_to_open: int = 0
    images_empty_or_invalid_label_file: int = 0
    images_needing_tiling: int = 0
    images_not_needing_tiling: int = 0
    images_with_no_retained_tiles: int = 0
    images_tiled_into_one_output: int = 0
    images_tiled_into_multiple_outputs: int = 0
    images_with_outputs: int = 0
    tiles_generated: int = 0
    tiles_kept_positive: int = 0
    tiles_kept_negative: int = 0
    tiles_discarded_negative: int = 0
    boxes_assigned_to_tiles: int = 0
    boxes_dropped_by_visible_threshold: int = 0
    boxes_clipped_to_tile_bounds: int = 0
    output_images_written: int = 0
    output_labels_written: int = 0
    total_input_boxes: int = 0
    total_output_boxes: int = 0
    malformed_label_rows: int = 0
    invalid_label_files: int = 0
    non_rectangle_objects: int = 0
    label_size_mismatches: int = 0


@dataclass(frozen=True)
class RunOptions:
    aspect_ratio: float
    delete_images_without_labels: bool
    tile_overlap_ratio: float
    tile_overlap_pixels: int | None
    negative_tile_keep_ratio: float
    assignment_mode: str
    min_visible_fraction: float


def split_dataset_on_x_axis(
    aspect_ratio: float,
    images_folder_path: str | Path,
    labels_folder_path: str | Path,
    output_images_folder_path: str | Path | None = None,
    output_labels_folder_path: str | Path | None = None,
    delete_images_without_labels: bool = False,
    logger: logging.Logger | None = None,
    workers: int = 4,
    tile_overlap_ratio: float = DEFAULT_TILE_OVERLAP_RATIO,
    tile_overlap_pixels: int | None = None,
    negative_tile_keep_ratio: float = DEFAULT_NEGATIVE_TILE_KEEP_RATIO,
    assignment_mode: str = DEFAULT_ASSIGNMENT_MODE,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> dict[str, Any]:
    """Tile a labeled dataset horizontally and write tiled images + Supervisely labels."""
    normalized_assignment_mode = _normalize_assignment_mode(assignment_mode)
    run_options = RunOptions(
        aspect_ratio=aspect_ratio,
        delete_images_without_labels=delete_images_without_labels,
        tile_overlap_ratio=tile_overlap_ratio,
        tile_overlap_pixels=tile_overlap_pixels,
        negative_tile_keep_ratio=negative_tile_keep_ratio,
        assignment_mode=normalized_assignment_mode,
        min_visible_fraction=min_visible_fraction,
    )
    _validate_run_options(run_options=run_options, workers=workers)

    images_dir = Path(images_folder_path)
    labels_dir = Path(labels_folder_path)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    ratio_token = f"{run_options.aspect_ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    output_images_dir = (
        Path(output_images_folder_path)
        if output_images_folder_path is not None
        else images_dir.parent / f"{images_dir.name}_split_r{ratio_token}"
    )
    output_labels_dir = (
        Path(output_labels_folder_path)
        if output_labels_folder_path is not None
        else labels_dir.parent / f"{labels_dir.name}_split_r{ratio_token}"
    )
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    active_logger = logger or _build_logger()
    stats = SplitStats(target_ratio=run_options.aspect_ratio)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    stats.total_images = len(image_paths)

    active_logger.info("Starting tiling run for %s images.", stats.total_images)
    active_logger.info("Target aspect ratio: %.6f", run_options.aspect_ratio)
    active_logger.info("Output images folder: %s", output_images_dir)
    active_logger.info("Output labels folder: %s", output_labels_dir)
    active_logger.info(
        "Tiling config: overlap_ratio=%.3f overlap_pixels=%s negative_keep_ratio=%.3f "
        "assignment_mode=%s min_visible_fraction=%.3f",
        run_options.tile_overlap_ratio,
        run_options.tile_overlap_pixels,
        run_options.negative_tile_keep_ratio,
        run_options.assignment_mode,
        run_options.min_visible_fraction,
    )

    if _output_dataset_is_populated(output_images_dir=output_images_dir, output_labels_dir=output_labels_dir):
        stats.runs_skipped_existing_outputs = 1
        active_logger.info(
            "Skipping tiling run (output dataset already populated): %s | %s",
            output_images_dir,
            output_labels_dir,
        )
        stats_dict = _build_stats_dict(
            stats=stats,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            workers=workers,
            run_options=run_options,
        )
        _write_stats_report(stats=stats_dict, output_images_dir=output_images_dir)
        return stats_dict

    if workers == 1:
        _process_images_sequential(
            image_paths=image_paths,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            run_options=run_options,
            logger=active_logger,
            stats=stats,
        )
    else:
        _process_images_parallel(
            image_paths=image_paths,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            workers=workers,
            run_options=run_options,
            logger=active_logger,
            stats=stats,
        )

    stats_dict = _build_stats_dict(
        stats=stats,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        workers=workers,
        run_options=run_options,
    )

    active_logger.info("Tiling run complete.")
    for key in sorted(stats_dict):
        active_logger.info("%s=%s", key, stats_dict[key])
    _write_stats_report(stats=stats_dict, output_images_dir=output_images_dir)

    return stats_dict


def _process_images_sequential(
    image_paths: list[Path],
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    run_options: RunOptions,
    logger: logging.Logger,
    stats: SplitStats,
) -> None:
    """Process images in the current process, updating aggregate stats in place."""
    for image_index, image_path in enumerate(image_paths, start=1):
        _print_progress(
            current=image_index,
            total=stats.total_images,
            image_name=image_path.name,
        )
        image_stats = _process_single_image(
            image_path=image_path,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            run_options=run_options,
            logger=logger,
        )
        _merge_split_stats(target=stats, delta=image_stats)


def _process_images_parallel(
    image_paths: list[Path],
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    workers: int,
    run_options: RunOptions,
    logger: logging.Logger,
    stats: SplitStats,
) -> None:
    """Process images with a worker pool and fallback to isolated retries on pool failure."""
    logger.info("Using %s worker processes.", workers)
    pending_paths: deque[Path] = deque(image_paths)
    future_to_path: dict[Any, Path] = {}
    completed_paths: set[Path] = set()
    retry_paths: list[Path] = []
    pool_failure: BrokenProcessPool | None = None
    completed_images = 0
    in_flight_limit = max(workers * DEFAULT_IN_FLIGHT_MULTIPLIER, workers)
    executor = _create_process_pool(max_workers=workers)
    try:
        def submit_more() -> None:
            while pending_paths and len(future_to_path) < in_flight_limit:
                image_path = pending_paths.popleft()
                future = executor.submit(
                    _process_single_image_worker,
                    image_path=str(image_path),
                    labels_dir=str(labels_dir),
                    output_images_dir=str(output_images_dir),
                    output_labels_dir=str(output_labels_dir),
                    run_options=run_options,
                )
                future_to_path[future] = image_path

        submit_more()

        while future_to_path:
            done_future = next(as_completed(list(future_to_path)))
            image_path = future_to_path.pop(done_future)
            completed_images += 1
            _print_progress(
                current=completed_images,
                total=stats.total_images,
                image_name=image_path.name,
            )
            try:
                image_stats_dict = done_future.result()
            except BrokenProcessPool as exc:
                pool_failure = exc
                retry_paths.append(image_path)
                logger.warning(
                    "Worker pool crashed while processing %s. "
                    "Switching remaining images to isolated worker retries.",
                    image_path.name,
                )
                break
            except Exception as exc:
                stats.images_failed_to_open += 1
                logger.warning("Worker failed for %s: %s", image_path.name, exc)
                completed_paths.add(image_path)
            else:
                _merge_split_stats(
                    target=stats,
                    delta=SplitStats(**image_stats_dict),
                )
                completed_paths.add(image_path)

            try:
                submit_more()
            except BrokenProcessPool as exc:
                pool_failure = exc
                logger.warning(
                    "Worker pool crashed while submitting new tasks. "
                    "Switching remaining images to isolated worker retries.",
                )
                break
    finally:
        if pool_failure is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    if pool_failure is None:
        return

    remaining_paths = retry_paths + list(future_to_path.values()) + list(pending_paths)
    completed_images = len(completed_paths)
    logger.warning(
        "Retrying %s remaining images in isolated worker mode after pool failure: %s",
        len(remaining_paths),
        pool_failure,
    )
    for image_path in remaining_paths:
        completed_images += 1
        _print_progress(
            current=completed_images,
            total=stats.total_images,
            image_name=image_path.name,
        )
        image_stats = _process_single_image_in_isolated_worker(
            image_path=image_path,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            run_options=run_options,
            logger=logger,
        )
        _merge_split_stats(target=stats, delta=image_stats)


def _process_single_image_worker(
    image_path: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    run_options: RunOptions,
) -> dict[str, Any]:
    image_stats = _process_single_image(
        image_path=Path(image_path),
        labels_dir=Path(labels_dir),
        output_images_dir=Path(output_images_dir),
        output_labels_dir=Path(output_labels_dir),
        run_options=run_options,
        logger=_build_null_logger(),
    )
    return asdict(image_stats)


def _create_process_pool(max_workers: int) -> ProcessPoolExecutor:
    try:
        return ProcessPoolExecutor(
            max_workers=max_workers,
            max_tasks_per_child=DEFAULT_MAX_TASKS_PER_CHILD,
        )
    except (TypeError, ValueError):
        return ProcessPoolExecutor(max_workers=max_workers)


def _process_single_image_in_isolated_worker(
    image_path: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    run_options: RunOptions,
    logger: logging.Logger,
) -> SplitStats:
    try:
        with _create_process_pool(max_workers=1) as executor:
            future = executor.submit(
                _process_single_image_worker,
                image_path=str(image_path),
                labels_dir=str(labels_dir),
                output_images_dir=str(output_images_dir),
                output_labels_dir=str(output_labels_dir),
                run_options=run_options,
            )
            image_stats_dict = future.result()
    except BrokenProcessPool as exc:
        stats = SplitStats(target_ratio=run_options.aspect_ratio)
        stats.images_failed_to_open += 1
        logger.warning("Isolated worker crashed for %s: %s", image_path.name, exc)
        return stats
    except Exception as exc:
        stats = SplitStats(target_ratio=run_options.aspect_ratio)
        stats.images_failed_to_open += 1
        logger.warning("Isolated worker failed for %s: %s", image_path.name, exc)
        return stats
    return SplitStats(**image_stats_dict)


def _process_single_image(
    image_path: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    run_options: RunOptions,
    logger: logging.Logger,
) -> SplitStats:
    """Process one image end-to-end: load, tile if needed, and write outputs."""
    stats = SplitStats(target_ratio=run_options.aspect_ratio)
    label_path = _find_label_path(image_path=image_path, labels_dir=labels_dir)
    if label_path is None:
        stats.images_missing_label_file += 1
        if run_options.delete_images_without_labels:
            try:
                image_path.unlink()
                stats.images_deleted_missing_label_file += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", image_path, exc)
        logger.info("Skipping %s (missing label file).", image_path.name)
        return stats

    image = _load_rgb_image(image_path=image_path, stats=stats, logger=logger)
    if image is None:
        return stats

    image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        stats.images_failed_to_open += 1
        logger.warning(
            "Skipping %s (invalid image size %sx%s).",
            image_path.name,
            image_width,
            image_height,
        )
        return stats

    boxes, label_metadata = _parse_supervisely_label_file(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
        stats=stats,
        logger=logger,
    )
    stats.images_with_label_file += 1
    stats.total_input_boxes += len(boxes)
    if not boxes:
        stats.images_empty_or_invalid_label_file += 1
        logger.info("Skipping %s (label file has no valid rectangle boxes).", image_path.name)
        return stats

    image_ratio = image_width / image_height
    if image_ratio <= run_options.aspect_ratio:
        stats.images_not_needing_tiling += 1
        _write_output_with_stats(
            image=image,
            boxes=boxes,
            label_metadata=label_metadata,
            output_image_path=output_images_dir / image_path.name,
            stats=stats,
            output_labels_dir=output_labels_dir,
        )
        stats.images_with_outputs += 1
        return stats

    stats.images_needing_tiling += 1
    tiled_segments = _tile_image_horizontally(
        image=image,
        boxes=boxes,
        target_ratio=run_options.aspect_ratio,
        overlap_ratio=run_options.tile_overlap_ratio,
        overlap_pixels=run_options.tile_overlap_pixels,
        image_key=image_path.name,
        negative_tile_keep_ratio=run_options.negative_tile_keep_ratio,
        assignment_mode=run_options.assignment_mode,
        min_visible_fraction=run_options.min_visible_fraction,
        stats=stats,
    )
    if not tiled_segments:
        stats.images_with_no_retained_tiles += 1
        logger.warning(
            "No tiles retained for %s; writing original image to avoid dropping data.",
            image_path.name,
        )
        _write_output_with_stats(
            image=image,
            boxes=boxes,
            label_metadata=label_metadata,
            output_image_path=output_images_dir / image_path.name,
            stats=stats,
            output_labels_dir=output_labels_dir,
        )
        stats.images_with_outputs += 1
        return stats

    written_outputs = 0
    output_box_count = 0
    for suffix, segment_image, segment_boxes in tiled_segments:
        output_image_name = f"{image_path.stem}_{suffix}{image_path.suffix}"
        _write_output_with_stats(
            image=segment_image,
            boxes=segment_boxes,
            label_metadata=label_metadata,
            output_image_path=output_images_dir / output_image_name,
            stats=stats,
            output_labels_dir=output_labels_dir,
        )

        written_outputs += 1
        output_box_count += len(segment_boxes)

    if run_options.assignment_mode == "center" and output_box_count != len(boxes):
        logger.warning(
            "Image %s did not preserve all boxes under center assignment (%s/%s boxes).",
            image_path.name,
            output_box_count,
            len(boxes),
        )

    if written_outputs == 0:
        stats.images_with_no_retained_tiles += 1
    elif written_outputs == 1:
        stats.images_tiled_into_one_output += 1
        stats.images_with_outputs += 1
    else:
        stats.images_tiled_into_multiple_outputs += 1
        stats.images_with_outputs += 1

    return stats


def _output_dataset_is_populated(
    output_images_dir: Path,
    output_labels_dir: Path,
) -> bool:
    has_any_output_image = any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in output_images_dir.iterdir()
    )
    has_any_output_label = any(
        path.is_file() and path.suffix.lower() == ".json" for path in output_labels_dir.iterdir()
    )
    return has_any_output_image and has_any_output_label


def _write_stats_report(stats: dict[str, Any], output_images_dir: Path) -> None:
    report_path = output_images_dir / "split_stats.txt"
    lines = [f"{key}={value}" for key, value in sorted(stats.items())]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_split_stats(target: SplitStats, delta: SplitStats) -> None:
    for field_name in SplitStats.__dataclass_fields__:
        if field_name in {"target_ratio", "total_images"}:
            continue
        setattr(target, field_name, getattr(target, field_name) + getattr(delta, field_name))


def _normalize_assignment_mode(assignment_mode: str) -> str:
    normalized_assignment_mode = assignment_mode.strip().lower()
    if normalized_assignment_mode in {"visible-area", "visible_area", "area"}:
        return "visible"
    return normalized_assignment_mode


def _validate_run_options(run_options: RunOptions, workers: int) -> None:
    if run_options.aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be a positive float (width / height).")
    if workers <= 0:
        raise ValueError("workers must be a positive integer.")
    if run_options.tile_overlap_ratio < 0 or run_options.tile_overlap_ratio >= 1:
        raise ValueError("tile_overlap_ratio must be in [0, 1).")
    if run_options.tile_overlap_pixels is not None and run_options.tile_overlap_pixels < 0:
        raise ValueError("tile_overlap_pixels must be >= 0 when provided.")
    if run_options.negative_tile_keep_ratio < 0 or run_options.negative_tile_keep_ratio > 1:
        raise ValueError("negative_tile_keep_ratio must be in [0, 1].")
    if run_options.min_visible_fraction < 0 or run_options.min_visible_fraction > 1:
        raise ValueError("min_visible_fraction must be in [0, 1].")
    if run_options.assignment_mode not in VALID_ASSIGNMENT_MODES:
        supported = ", ".join(sorted(VALID_ASSIGNMENT_MODES))
        raise ValueError(f"assignment_mode must be one of: {supported}")


def _build_stats_dict(
    stats: SplitStats,
    output_images_dir: Path,
    output_labels_dir: Path,
    workers: int,
    run_options: RunOptions,
) -> dict[str, Any]:
    stats_dict: dict[str, Any] = asdict(stats)
    stats_dict["output_images_folder"] = str(output_images_dir)
    stats_dict["output_labels_folder"] = str(output_labels_dir)
    stats_dict["delete_images_without_labels"] = run_options.delete_images_without_labels
    stats_dict["workers"] = workers
    stats_dict["tile_overlap_ratio"] = run_options.tile_overlap_ratio
    stats_dict["tile_overlap_pixels"] = run_options.tile_overlap_pixels
    stats_dict["negative_tile_keep_ratio"] = run_options.negative_tile_keep_ratio
    stats_dict["assignment_mode"] = run_options.assignment_mode
    stats_dict["min_visible_fraction"] = run_options.min_visible_fraction
    return stats_dict


def _write_output_with_stats(
    image: Image.Image,
    boxes: list[Box],
    label_metadata: dict[str, Any],
    output_image_path: Path,
    stats: SplitStats,
    output_labels_dir: Path,
) -> None:
    output_label_path = output_labels_dir / f"{output_image_path.name}.json"
    _write_output_pair(
        image=image,
        boxes=boxes,
        label_metadata=label_metadata,
        output_image_path=output_image_path,
        output_label_path=output_label_path,
    )
    stats.output_images_written += 1
    stats.output_labels_written += 1
    stats.total_output_boxes += len(boxes)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _build_null_logger() -> logging.Logger:
    logger = logging.getLogger(f"{LOGGER_NAME}.null")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


def _print_progress(current: int, total: int, image_name: str) -> None:
    if total <= 0:
        return
    percent = (current / total) * 100.0
    end = "\n" if current >= total else ""
    # Keep this concise for quick terminal progress checks.
    print(
        f"\rProgress {current}/{total} ({percent:5.1f}%) - {image_name}",
        end=end,
        flush=True,
    )


def _find_label_path(image_path: Path, labels_dir: Path) -> Path | None:
    for candidate in (
        labels_dir / f"{image_path.name}.json",
        labels_dir / f"{image_path.stem}.json",
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_rgb_image(
    image_path: Path,
    stats: SplitStats,
    logger: logging.Logger,
) -> Image.Image | None:
    try:
        with Image.open(image_path) as source_image:
            return source_image.convert("RGB")
    except OSError as exc:
        stats.images_failed_to_open += 1
        logger.warning("Skipping %s (cannot open image): %s", image_path.name, exc)
        return None


def _parse_supervisely_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    stats: SplitStats,
    logger: logging.Logger,
) -> tuple[list[Box], dict[str, Any]]:
    try:
        raw_data = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stats.invalid_label_files += 1
        logger.warning("Failed to parse label file %s: %s", label_path, exc)
        return [], {}

    if not isinstance(raw_data, dict):
        stats.invalid_label_files += 1
        logger.warning("Label file is not a JSON object: %s", label_path)
        return [], {}

    _track_label_size_mismatch(
        raw_data=raw_data,
        image_width=image_width,
        image_height=image_height,
        stats=stats,
        logger=logger,
        label_path=label_path,
    )

    objects = raw_data.get("objects")
    if not isinstance(objects, list):
        stats.invalid_label_files += 1
        logger.warning("Label file missing valid 'objects' array: %s", label_path)
        return [], _extract_label_metadata(raw_data)

    boxes: list[Box] = []
    for idx, obj in enumerate(objects):
        box = _object_to_box(
            obj=obj,
            image_width=image_width,
            image_height=image_height,
            stats=stats,
            logger=logger,
            label_name=label_path.name,
            object_index=idx,
        )
        if box is not None:
            boxes.append(box)
    return boxes, _extract_label_metadata(raw_data)


def _extract_label_metadata(raw_data: dict[str, Any]) -> dict[str, Any]:
    metadata = deepcopy(raw_data)
    metadata.pop("objects", None)
    metadata.pop("size", None)
    return metadata


def _track_label_size_mismatch(
    raw_data: dict[str, Any],
    image_width: int,
    image_height: int,
    stats: SplitStats,
    logger: logging.Logger,
    label_path: Path,
) -> None:
    size = raw_data.get("size")
    if not isinstance(size, dict):
        return

    label_width = size.get("width")
    label_height = size.get("height")
    if not isinstance(label_width, (int, float)) or not isinstance(label_height, (int, float)):
        return

    if int(label_width) != image_width or int(label_height) != image_height:
        stats.label_size_mismatches += 1
        logger.warning(
            "Size mismatch in %s (label=%sx%s image=%sx%s).",
            label_path.name,
            label_width,
            label_height,
            image_width,
            image_height,
        )


def _object_to_box(
    obj: Any,
    image_width: int,
    image_height: int,
    stats: SplitStats,
    logger: logging.Logger,
    label_name: str,
    object_index: int,
) -> Box | None:
    if not isinstance(obj, dict):
        stats.malformed_label_rows += 1
        logger.warning("Malformed object %s in %s (not an object).", object_index, label_name)
        return None

    if obj.get("geometryType") != "rectangle":
        stats.non_rectangle_objects += 1
        return None

    points = obj.get("points")
    if not isinstance(points, dict):
        stats.malformed_label_rows += 1
        logger.warning("Malformed object %s in %s (missing points).", object_index, label_name)
        return None

    exterior = points.get("exterior")
    if not isinstance(exterior, list) or len(exterior) != 2:
        stats.malformed_label_rows += 1
        logger.warning("Malformed object %s in %s (invalid exterior).", object_index, label_name)
        return None

    p1 = _parse_point(exterior[0])
    p2 = _parse_point(exterior[1])
    if p1 is None or p2 is None:
        stats.malformed_label_rows += 1
        logger.warning("Malformed object %s in %s (invalid exterior coordinates).", object_index, label_name)
        return None

    x1, y1 = p1
    x2, y2 = p2
    x_min = max(0.0, min(float(image_width), min(x1, x2)))
    x_max = max(0.0, min(float(image_width), max(x1, x2)))
    y_min = max(0.0, min(float(image_height), min(y1, y2)))
    y_max = max(0.0, min(float(image_height), max(y1, y2)))
    if x_max <= x_min or y_max <= y_min:
        stats.malformed_label_rows += 1
        logger.warning("Malformed object %s in %s (degenerate box).", object_index, label_name)
        return None

    return Box(
        class_title=str(obj.get("classTitle", "cone")),
        class_id=_parse_class_id(obj.get("classId")),
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        source_object=deepcopy(obj),
    )


def _parse_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def _parse_class_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _tile_image_horizontally(
    image: Image.Image,
    boxes: list[Box],
    target_ratio: float,
    overlap_ratio: float,
    overlap_pixels: int | None,
    image_key: str,
    negative_tile_keep_ratio: float,
    assignment_mode: str,
    min_visible_fraction: float,
    stats: SplitStats,
) -> list[tuple[str, Image.Image, list[Box]]]:
    """Create deterministic horizontal tiles and assign boxes to each retained tile."""
    image_width, image_height = image.size
    tile_width = _compute_tile_width(
        image_height=image_height,
        target_ratio=target_ratio,
        image_width=image_width,
    )
    tile_ranges = _generate_tile_x_ranges(
        image_width=image_width,
        tile_width=tile_width,
        overlap_ratio=overlap_ratio,
        overlap_pixels=overlap_pixels,
    )
    stats.tiles_generated += len(tile_ranges)

    center_owner_by_box_index: dict[int, int] | None = None
    if assignment_mode == "center":
        center_owner_by_box_index = _compute_center_owner_map(
            boxes=boxes,
            tile_ranges=tile_ranges,
        )

    kept_segments: list[tuple[str, Image.Image, list[Box]]] = []
    for tile_index, (x_start, x_end) in enumerate(tile_ranges):
        tile_boxes = _assign_boxes_to_tile(
            boxes=boxes,
            tile_ranges=tile_ranges,
            tile_index=tile_index,
            assignment_mode=assignment_mode,
            min_visible_fraction=min_visible_fraction,
            center_owner_by_box_index=center_owner_by_box_index,
            stats=stats,
        )
        is_positive_tile = bool(tile_boxes)
        if is_positive_tile:
            stats.tiles_kept_positive += 1
        else:
            keep_negative = _should_keep_negative_tile(
                image_key=image_key,
                tile_index=tile_index,
                x_start=x_start,
                x_end=x_end,
                keep_ratio=negative_tile_keep_ratio,
            )
            if keep_negative:
                stats.tiles_kept_negative += 1
            else:
                stats.tiles_discarded_negative += 1
                continue

        tile_suffix = _build_tile_suffix(tile_index=tile_index)
        tile_image = image.crop((x_start, 0, x_end, image_height))
        kept_segments.append((tile_suffix, tile_image, tile_boxes))

    return kept_segments


def _compute_tile_width(image_height: int, target_ratio: float, image_width: int) -> int:
    tile_width = int(round(target_ratio * image_height))
    tile_width = max(tile_width, 1)
    tile_width = min(tile_width, image_width)
    return tile_width


def _generate_tile_x_ranges(
    image_width: int,
    tile_width: int,
    overlap_ratio: float,
    overlap_pixels: int | None,
) -> list[tuple[int, int]]:
    if image_width <= 0:
        return []
    if tile_width >= image_width:
        return [(0, image_width)]

    if overlap_pixels is None:
        overlap = int(round(tile_width * overlap_ratio))
    else:
        overlap = int(round(overlap_pixels))
    overlap = max(0, min(overlap, tile_width - 1))
    stride = max(tile_width - overlap, 1)

    max_start = image_width - tile_width
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)

    tile_ranges: list[tuple[int, int]] = []
    for x_start in starts:
        x_end = min(x_start + tile_width, image_width)
        tile_ranges.append((x_start, x_end))
    return tile_ranges


def _compute_center_owner_map(
    boxes: list[Box],
    tile_ranges: list[tuple[int, int]],
) -> dict[int, int]:
    owner_map: dict[int, int] = {}
    if not tile_ranges:
        return owner_map

    last_tile_index = len(tile_ranges) - 1
    for box_index, box in enumerate(boxes):
        center_x = 0.5 * (box.x_min + box.x_max)
        candidate_indices: list[int] = []
        for tile_index, (x_start, x_end) in enumerate(tile_ranges):
            is_last_tile = tile_index == last_tile_index
            if _center_in_tile(center_x=center_x, x_start=x_start, x_end=x_end, is_last_tile=is_last_tile):
                candidate_indices.append(tile_index)

        if candidate_indices:
            owner_tile_index = min(
                candidate_indices,
                key=lambda idx: (
                    abs(_tile_center_x(tile_ranges[idx]) - center_x),
                    idx,
                ),
            )
        else:
            owner_tile_index = min(
                range(len(tile_ranges)),
                key=lambda idx: (
                    _distance_to_tile(center_x=center_x, tile_range=tile_ranges[idx]),
                    idx,
                ),
            )
        owner_map[box_index] = owner_tile_index

    return owner_map


def _center_in_tile(center_x: float, x_start: int, x_end: int, is_last_tile: bool) -> bool:
    if is_last_tile:
        return x_start <= center_x <= x_end
    return x_start <= center_x < x_end


def _tile_center_x(tile_range: tuple[int, int]) -> float:
    x_start, x_end = tile_range
    return 0.5 * float(x_start + x_end)


def _distance_to_tile(center_x: float, tile_range: tuple[int, int]) -> float:
    x_start, x_end = tile_range
    if center_x < x_start:
        return float(x_start) - center_x
    if center_x > x_end:
        return center_x - float(x_end)
    return 0.0


def _assign_boxes_to_tile(
    boxes: list[Box],
    tile_ranges: list[tuple[int, int]],
    tile_index: int,
    assignment_mode: str,
    min_visible_fraction: float,
    center_owner_by_box_index: dict[int, int] | None,
    stats: SplitStats,
) -> list[Box]:
    x_start, x_end = tile_ranges[tile_index]
    tile_boxes: list[Box] = []

    for box_index, box in enumerate(boxes):
        include_box = False
        if assignment_mode == "center":
            if center_owner_by_box_index is not None:
                include_box = center_owner_by_box_index.get(box_index) == tile_index
        else:
            visible_fraction = _visible_area_fraction_in_tile(
                box=box,
                x_start=x_start,
                x_end=x_end,
            )
            include_box = visible_fraction >= min_visible_fraction
            if not include_box and visible_fraction > 0:
                stats.boxes_dropped_by_visible_threshold += 1

        if not include_box:
            continue

        clipped_box, was_clipped = _clip_box_to_tile_bounds(
            box=box,
            x_start=x_start,
            x_end=x_end,
        )
        if clipped_box is None:
            continue

        tile_boxes.append(clipped_box.shifted(dx=-x_start))
        stats.boxes_assigned_to_tiles += 1
        if was_clipped:
            stats.boxes_clipped_to_tile_bounds += 1

    return tile_boxes


def _visible_area_fraction_in_tile(box: Box, x_start: int, x_end: int) -> float:
    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min
    if box_width <= 0 or box_height <= 0:
        return 0.0

    intersect_x_min = max(box.x_min, float(x_start))
    intersect_x_max = min(box.x_max, float(x_end))
    intersect_width = max(0.0, intersect_x_max - intersect_x_min)
    if intersect_width <= 0:
        return 0.0

    intersection_area = intersect_width * box_height
    box_area = box_width * box_height
    return intersection_area / box_area


def _clip_box_to_tile_bounds(box: Box, x_start: int, x_end: int) -> tuple[Box | None, bool]:
    clipped_x_min = max(box.x_min, float(x_start))
    clipped_x_max = min(box.x_max, float(x_end))
    if clipped_x_max <= clipped_x_min:
        return None, False

    clipped_box = Box(
        class_title=box.class_title,
        class_id=box.class_id,
        x_min=clipped_x_min,
        y_min=box.y_min,
        x_max=clipped_x_max,
        y_max=box.y_max,
        source_object=box.source_object,
    )
    was_clipped = (
        clipped_box.x_min != box.x_min
        or clipped_box.x_max != box.x_max
        or clipped_box.y_min != box.y_min
        or clipped_box.y_max != box.y_max
    )
    return clipped_box, was_clipped


def _should_keep_negative_tile(
    image_key: str,
    tile_index: int,
    x_start: int,
    x_end: int,
    keep_ratio: float,
) -> bool:
    if keep_ratio <= 0:
        return False
    if keep_ratio >= 1:
        return True
    token = f"{image_key}|{tile_index}|{x_start}|{x_end}".encode("utf-8")
    sample_bytes = hashlib.blake2b(token, digest_size=8).digest()
    sample_value = int.from_bytes(sample_bytes, byteorder="big", signed=False) / float(2**64)
    return sample_value < keep_ratio


def _build_tile_suffix(tile_index: int) -> str:
    return f"tile{tile_index:03d}"


def _write_output_pair(
    image: Image.Image,
    boxes: list[Box],
    label_metadata: dict[str, Any],
    output_image_path: Path,
    output_label_path: Path,
) -> None:
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path)
    _write_supervisely_label_file(
        label_path=output_label_path,
        boxes=boxes,
        image_width=image.width,
        image_height=image.height,
        label_metadata=label_metadata,
    )


def _write_supervisely_label_file(
    label_path: Path,
    boxes: list[Box],
    image_width: int,
    image_height: int,
    label_metadata: dict[str, Any],
) -> None:
    payload = deepcopy(label_metadata) if isinstance(label_metadata, dict) else {}
    payload["size"] = {"height": image_height, "width": image_width}
    payload["objects"] = [
        _box_to_supervisely_object(box=box, image_width=image_width, image_height=image_height)
        for box in boxes
    ]
    label_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _box_to_supervisely_object(box: Box, image_width: int, image_height: int) -> dict[str, Any]:
    obj = deepcopy(box.source_object) if isinstance(box.source_object, dict) else {}
    points = obj.get("points")
    if not isinstance(points, dict):
        points = {}
    obj["points"] = points

    points["exterior"] = [
        [_clip_coord(box.x_min, image_width), _clip_coord(box.y_min, image_height)],
        [_clip_coord(box.x_max, image_width), _clip_coord(box.y_max, image_height)],
    ]
    points["interior"] = []
    obj["geometryType"] = "rectangle"
    obj["classTitle"] = box.class_title
    if box.class_id is not None:
        obj["classId"] = box.class_id
    return obj


def _clip_coord(value: float, max_value: int) -> int:
    return int(round(min(max(value, 0.0), float(max_value))))


def main() -> None:
    # Set defaults here for direct script execution (no CLI parser).
    images_path = DEFAULT_IMAGES_DIR
    labels_path = DEFAULT_LABELS_DIR
    output_images_path: Path | None = None
    output_labels_path: Path | None = None
    delete_images_without_labels = False
    workers = 1
    aspect_ratio = DEFAULT_ASPECT_RATIO
    tile_overlap_ratio = DEFAULT_TILE_OVERLAP_RATIO
    tile_overlap_pixels: int | None = None
    negative_tile_keep_ratio = DEFAULT_NEGATIVE_TILE_KEEP_RATIO
    assignment_mode = DEFAULT_ASSIGNMENT_MODE
    min_visible_fraction = DEFAULT_MIN_VISIBLE_FRACTION

    stats = split_dataset_on_x_axis(
        aspect_ratio=aspect_ratio,
        images_folder_path=images_path,
        labels_folder_path=labels_path,
        output_images_folder_path=output_images_path,
        output_labels_folder_path=output_labels_path,
        delete_images_without_labels=delete_images_without_labels,
        workers=workers,
        tile_overlap_ratio=tile_overlap_ratio,
        tile_overlap_pixels=tile_overlap_pixels,
        negative_tile_keep_ratio=negative_tile_keep_ratio,
        assignment_mode=assignment_mode,
        min_visible_fraction=min_visible_fraction,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
