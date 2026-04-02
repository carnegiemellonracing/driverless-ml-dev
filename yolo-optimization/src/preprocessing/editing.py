from __future__ import annotations

import argparse
import json
import logging
from bisect import bisect_left, bisect_right
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
DEFAULT_IMAGES_DIR = Path("ml_data") / "" / "ampera" / "img"
DEFAULT_LABELS_DIR = Path("ml_data") / "fsoco_bounding_boxes_train" / "ampera" / "ann"
DEFAULT_ASPECT_RATIO = 1
DEFAULT_MAX_TASKS_PER_CHILD = 128
DEFAULT_IN_FLIGHT_MULTIPLIER = 4


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
    images_needing_split: int = 0
    images_not_needing_split: int = 0
    images_unsplittable: int = 0
    images_split_selected: int = 0
    images_split_into_one_output: int = 0
    images_split_into_two_outputs: int = 0
    images_with_outputs: int = 0
    split_outputs_discarded_no_cones: int = 0
    split_candidates_total: int = 0
    split_candidates_cut_box: int = 0
    split_candidates_valid: int = 0
    split_candidates_with_empty_side: int = 0
    split_candidates_with_two_cone_sides: int = 0
    selected_splits_with_empty_side: int = 0
    output_images_written: int = 0
    output_labels_written: int = 0
    total_input_boxes: int = 0
    total_output_boxes: int = 0
    malformed_label_rows: int = 0
    invalid_label_files: int = 0
    non_rectangle_objects: int = 0
    label_size_mismatches: int = 0


def split_dataset_on_x_axis(
    aspect_ratio: float,
    images_folder_path: str | Path,
    labels_folder_path: str | Path,
    output_images_folder_path: str | Path | None = None,
    output_labels_folder_path: str | Path | None = None,
    delete_images_without_labels: bool = False,
    logger: logging.Logger | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be a positive float (width / height).")
    if workers <= 0:
        raise ValueError("workers must be a positive integer.")

    images_dir = Path(images_folder_path)
    labels_dir = Path(labels_folder_path)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    ratio_token = f"{aspect_ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")
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
    stats = SplitStats(target_ratio=aspect_ratio)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    stats.total_images = len(image_paths)

    active_logger.info("Starting split run for %s images.", stats.total_images)
    active_logger.info("Target aspect ratio: %.6f", aspect_ratio)
    active_logger.info("Output images folder: %s", output_images_dir)
    active_logger.info("Output labels folder: %s", output_labels_dir)

    if _output_dataset_is_populated(output_images_dir=output_images_dir, output_labels_dir=output_labels_dir):
        stats.runs_skipped_existing_outputs = 1
        active_logger.info(
            "Skipping split run (output dataset already populated): %s | %s",
            output_images_dir,
            output_labels_dir,
        )
        stats_dict: dict[str, Any] = asdict(stats)
        stats_dict["output_images_folder"] = str(output_images_dir)
        stats_dict["output_labels_folder"] = str(output_labels_dir)
        stats_dict["delete_images_without_labels"] = delete_images_without_labels
        stats_dict["workers"] = workers
        _write_stats_report(stats=stats_dict, output_images_dir=output_images_dir)
        return stats_dict

    if workers == 1:
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
                aspect_ratio=aspect_ratio,
                delete_images_without_labels=delete_images_without_labels,
                logger=active_logger,
            )
            _merge_split_stats(target=stats, delta=image_stats)
    else:
        active_logger.info("Using %s worker processes.", workers)
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
                        aspect_ratio=aspect_ratio,
                        delete_images_without_labels=delete_images_without_labels,
                    )
                    future_to_path[future] = image_path

            submit_more()

            while future_to_path:
                done_future = next(as_completed(list(future_to_path)))
                image_path = future_to_path.pop(done_future)
                image_name = image_path.name
                completed_images += 1
                _print_progress(
                    current=completed_images,
                    total=stats.total_images,
                    image_name=image_name,
                )
                try:
                    image_stats_dict = done_future.result()
                except BrokenProcessPool as exc:
                    pool_failure = exc
                    retry_paths.append(image_path)
                    active_logger.warning(
                        "Worker pool crashed while processing %s. "
                        "Switching remaining images to isolated worker retries.",
                        image_name,
                    )
                    break
                except Exception as exc:
                    stats.images_failed_to_open += 1
                    active_logger.warning("Worker failed for %s: %s", image_name, exc)
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
                    active_logger.warning(
                        "Worker pool crashed while submitting new tasks. "
                        "Switching remaining images to isolated worker retries.",
                    )
                    break
        finally:
            if pool_failure is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        if pool_failure is not None:
            remaining_paths = retry_paths + list(future_to_path.values()) + list(pending_paths)
            completed_images = len(completed_paths)
            active_logger.warning(
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
                    aspect_ratio=aspect_ratio,
                    delete_images_without_labels=delete_images_without_labels,
                    logger=active_logger,
                )
                _merge_split_stats(target=stats, delta=image_stats)

    stats_dict: dict[str, Any] = asdict(stats)
    stats_dict["output_images_folder"] = str(output_images_dir)
    stats_dict["output_labels_folder"] = str(output_labels_dir)
    stats_dict["delete_images_without_labels"] = delete_images_without_labels
    stats_dict["workers"] = workers

    active_logger.info("Split run complete.")
    for key in sorted(stats_dict):
        active_logger.info("%s=%s", key, stats_dict[key])
    _write_stats_report(stats=stats_dict, output_images_dir=output_images_dir)

    return stats_dict


def _process_single_image_worker(
    image_path: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    aspect_ratio: float,
    delete_images_without_labels: bool,
) -> dict[str, Any]:
    image_stats = _process_single_image(
        image_path=Path(image_path),
        labels_dir=Path(labels_dir),
        output_images_dir=Path(output_images_dir),
        output_labels_dir=Path(output_labels_dir),
        aspect_ratio=aspect_ratio,
        delete_images_without_labels=delete_images_without_labels,
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
    aspect_ratio: float,
    delete_images_without_labels: bool,
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
                aspect_ratio=aspect_ratio,
                delete_images_without_labels=delete_images_without_labels,
            )
            image_stats_dict = future.result()
    except BrokenProcessPool as exc:
        stats = SplitStats(target_ratio=aspect_ratio)
        stats.images_failed_to_open += 1
        logger.warning("Isolated worker crashed for %s: %s", image_path.name, exc)
        return stats
    except Exception as exc:
        stats = SplitStats(target_ratio=aspect_ratio)
        stats.images_failed_to_open += 1
        logger.warning("Isolated worker failed for %s: %s", image_path.name, exc)
        return stats
    return SplitStats(**image_stats_dict)


def _process_single_image(
    image_path: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    aspect_ratio: float,
    delete_images_without_labels: bool,
    logger: logging.Logger,
) -> SplitStats:
    stats = SplitStats(target_ratio=aspect_ratio)
    label_path = _find_label_path(image_path=image_path, labels_dir=labels_dir)
    if label_path is None:
        stats.images_missing_label_file += 1
        if delete_images_without_labels:
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
    if image_ratio <= aspect_ratio:
        stats.images_not_needing_split += 1
        output_image_path = output_images_dir / image_path.name
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
        stats.images_with_outputs += 1
        return stats

    stats.images_needing_split += 1
    split_segments = _split_segments_until_target_ratio(
        image=image,
        boxes=boxes,
        target_ratio=aspect_ratio,
        stats=stats,
    )
    if not split_segments:
        stats.images_unsplittable += 1
        logger.info("Skipping %s (no valid split boundary found).", image_path.name)
        return stats

    written_outputs = 0
    output_box_count = 0
    for suffix, segment_image, segment_boxes in split_segments:
        output_image_name = f"{image_path.stem}_{suffix}{image_path.suffix}"
        output_image_path = output_images_dir / output_image_name
        output_label_path = output_labels_dir / f"{output_image_name}.json"
        _write_output_pair(
            image=segment_image,
            boxes=segment_boxes,
            label_metadata=label_metadata,
            output_image_path=output_image_path,
            output_label_path=output_label_path,
        )

        written_outputs += 1
        output_box_count += len(segment_boxes)
        stats.output_images_written += 1
        stats.output_labels_written += 1
        stats.total_output_boxes += len(segment_boxes)

    if output_box_count != len(boxes):
        logger.warning(
            "Image %s did not preserve all cones after split (%s/%s boxes).",
            image_path.name,
            output_box_count,
            len(boxes),
        )

    if written_outputs == 0:
        stats.images_unsplittable += 1
    elif written_outputs == 1:
        stats.images_split_into_one_output += 1
        stats.images_with_outputs += 1
    else:
        stats.images_split_into_two_outputs += 1
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


def _choose_best_split_boundary(
    boxes: list[Box],
    image_width: int,
    image_height: int,
    target_ratio: float,
    stats: SplitStats,
) -> int | None:
    if image_width <= 1 or not boxes:
        return None

    box_count = len(boxes)
    sorted_x_max = sorted(box.x_max for box in boxes)
    sorted_x_min = sorted(box.x_min for box in boxes)

    best_boundary: int | None = None
    best_score: tuple[float, float, float, float] | None = None
    stats.split_candidates_total += image_width - 1

    for boundary in range(1, image_width):
        left_count = bisect_right(sorted_x_max, boundary)
        right_count = box_count - bisect_left(sorted_x_min, boundary)
        if left_count + right_count != box_count:
            stats.split_candidates_cut_box += 1
            continue

        stats.split_candidates_valid += 1
        empty_side_count = int(left_count == 0) + int(right_count == 0)
        if empty_side_count > 0:
            stats.split_candidates_with_empty_side += 1
        else:
            stats.split_candidates_with_two_cone_sides += 1

        left_error = abs((boundary / image_height) - target_ratio)
        right_error = abs(((image_width - boundary) / image_height) - target_ratio)
        score = (
            float(empty_side_count),
            left_error + right_error,
            max(left_error, right_error),
            abs(boundary - (image_width - boundary)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_boundary = boundary
    return best_boundary


def _count_boxes_for_boundary(boxes: list[Box], boundary: int) -> tuple[int, int, bool]:
    left_count = 0
    right_count = 0
    for box in boxes:
        if box.x_max <= boundary:
            left_count += 1
        elif box.x_min >= boundary:
            right_count += 1
        else:
            return 0, 0, True
    return left_count, right_count, False


def _split_segments_until_target_ratio(
    image: Image.Image,
    boxes: list[Box],
    target_ratio: float,
    stats: SplitStats,
) -> list[tuple[str, Image.Image, list[Box]]]:
    pending_segments: deque[tuple[str, Image.Image, list[Box]]] = deque()
    pending_segments.append(("", image, boxes))
    ready_segments: list[tuple[str, Image.Image, list[Box]]] = []

    while pending_segments:
        parent_suffix, segment_image, segment_boxes = pending_segments.popleft()
        segment_width, segment_height = segment_image.size
        if segment_width <= 0 or segment_height <= 0:
            continue

        segment_ratio = segment_width / segment_height
        if segment_ratio <= target_ratio:
            ready_segments.append((parent_suffix, segment_image, segment_boxes))
            continue

        split_boundary = _choose_best_split_boundary(
            boxes=segment_boxes,
            image_width=segment_width,
            image_height=segment_height,
            target_ratio=target_ratio,
            stats=stats,
        )
        if split_boundary is None:
            continue

        stats.images_split_selected += 1
        left_count, right_count = _count_boxes_for_boundary(boxes=segment_boxes, boundary=split_boundary)[:2]
        if left_count == 0 or right_count == 0:
            stats.selected_splits_with_empty_side += 1

        for child_index, x_start, x_end in (
            (0, 0, split_boundary),
            (1, split_boundary, segment_width),
        ):
            child_boxes = _collect_segment_boxes(boxes=segment_boxes, x_start=x_start, x_end=x_end)
            if not child_boxes:
                stats.split_outputs_discarded_no_cones += 1
                continue

            child_suffix = _build_child_suffix(parent_suffix=parent_suffix, child_index=child_index)
            child_image = segment_image.crop((x_start, 0, x_end, segment_height))
            pending_segments.append((child_suffix, child_image, child_boxes))

    return ready_segments


def _build_child_suffix(parent_suffix: str, child_index: int) -> str:
    if not parent_suffix:
        return f"split{child_index}"
    return f"{parent_suffix}_{child_index}"


def _collect_segment_boxes(boxes: list[Box], x_start: int, x_end: int) -> list[Box]:
    segment_boxes: list[Box] = []
    for box in boxes:
        if box.x_max <= x_start or box.x_min >= x_end:
            continue
        if box.x_min < x_start or box.x_max > x_end:
            return []
        segment_boxes.append(box.shifted(dx=-x_start))
    return segment_boxes


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split dataset images on X-axis using Supervisely JSON labels.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Path to image folder (default: {DEFAULT_IMAGES_DIR}).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help=f"Path to label folder (default: {DEFAULT_LABELS_DIR}).",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=DEFAULT_ASPECT_RATIO,
        help=f"Target aspect ratio as width/height float (default: {DEFAULT_ASPECT_RATIO:.6f}).",
    )
    parser.add_argument("--output-images", type=Path, default=None, help="Optional output image folder.")
    parser.add_argument("--output-labels", type=Path, default=None, help="Optional output label folder.")
    parser.add_argument(
        "--delete-images-without-labels",
        action="store_true",
        help="Delete source images that do not have matching label files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-image processing (default: 1).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.aspect_ratio <= 0:
        parser.error("--aspect-ratio must be a positive float (width / height).")
    if args.workers <= 0:
        parser.error("--workers must be a positive integer.")

    stats = split_dataset_on_x_axis(
        aspect_ratio=args.aspect_ratio,
        images_folder_path=args.images,
        labels_folder_path=args.labels,
        output_images_folder_path=args.output_images,
        output_labels_folder_path=args.output_labels,
        delete_images_without_labels=args.delete_images_without_labels,
        workers=args.workers,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
