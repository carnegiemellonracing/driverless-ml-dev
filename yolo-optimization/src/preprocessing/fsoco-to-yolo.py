import argparse
import shutil
from pathlib import Path
from typing import Iterable


def _clear_directory(path: Path) -> None:
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists() or not path.is_dir():
        return ()
    return (p for p in path.iterdir() if p.is_file())


def _count_files(path: Path) -> int:
    return sum(1 for _ in _iter_files(path))


def copy_and_flatten_dataset(
    source_path: Path,
    dest_path: Path,
    *,
    overwrite: bool = False,
    prefix_with_team: bool = True,
) -> dict[str, int]:
    """
    Flatten a raw FSOCO dataset layout:

    source/
      - meta.json
      - teamA/{ann,img}
      - teamB/{ann,img}
      ...
    ->
    dest/
      - ann/
      - img/
      - meta.json
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    if not source_path.exists() or not source_path.is_dir() or not any(source_path.iterdir()):
        raise FileNotFoundError(f"Source dataset not found or empty: {source_path}")

    if dest_path.exists() and any(dest_path.iterdir()) and not overwrite:
        ann_count = _count_files(dest_path / "ann")
        img_count = _count_files(dest_path / "img")
        return {
            "teams": 0,
            "ann_files": ann_count,
            "img_files": img_count,
            "skipped_existing_outputs": 1,
        }

    dest_path.mkdir(parents=True, exist_ok=True)
    _clear_directory(dest_path)

    ann_out = dest_path / "ann"
    img_out = dest_path / "img"
    ann_out.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    meta_src = source_path / "meta.json"
    if meta_src.exists() and meta_src.is_file():
        shutil.copy2(meta_src, dest_path / "meta.json")

    ann_count = 0
    img_count = 0
    team_count = 0

    for team_dir in source_path.iterdir():
        if not team_dir.is_dir() or team_dir.name in {"ann", "img"}:
            continue

        team_count += 1
        ann_dir = team_dir / "ann"
        img_dir = team_dir / "img"

        for ann_file in _iter_files(ann_dir):
            out_name = f"{team_dir.name}_{ann_file.name}" if prefix_with_team else ann_file.name
            shutil.copy2(ann_file, ann_out / out_name)
            ann_count += 1

        for img_file in _iter_files(img_dir):
            out_name = f"{team_dir.name}_{img_file.name}" if prefix_with_team else img_file.name
            shutil.copy2(img_file, img_out / out_name)
            img_count += 1

    return {
        "teams": team_count,
        "ann_files": ann_count,
        "img_files": img_count,
        "skipped_existing_outputs": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flatten FSOCO source dataset into ann/img folders.")
    parser.add_argument("source", type=Path, help="Path to source dataset directory.")
    parser.add_argument("dest", type=Path, help="Path to destination directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already has contents.",
    )
    parser.add_argument(
        "--no-team-prefix",
        action="store_true",
        help="Do not prefix output files with team directory name.",
    )

    args = parser.parse_args(argv)

    stats = copy_and_flatten_dataset(
        args.source,
        args.dest,
        overwrite=args.overwrite,
        prefix_with_team=not args.no_team_prefix,
    )

    if stats["skipped_existing_outputs"]:
        print(
            "Skipping copy+flatten: destination already populated: "
            f"{args.dest}. Existing files - "
            f"{args.dest / 'ann'} ({stats['ann_files']}), "
            f"{args.dest / 'img'} ({stats['img_files']})."
        )
    else:
        print(
            "Copy+flatten complete: "
            f"{args.dest / 'ann'} ({stats['ann_files']} files), "
            f"{args.dest / 'img'} ({stats['img_files']} files), "
            f"teams processed: {stats['teams']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
