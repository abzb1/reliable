import argparse
import json
import os
from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict, Any

from nltk import edit_distance
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm  # progress bar

###############################################################
# Utility ----------------------------------------------------#
###############################################################

def _process_one(args: Tuple[int, str, Path]) -> Tuple[int, Dict[str, Any]]:
    """Worker function executed in a separate process.

    Parameters
    ----------
    args : Tuple
        (index, ground_truth_json_str, generation_dir)

    Returns
    -------
    Tuple[int, Dict[str, Any]]
        The original sample index and its evaluation dictionary.
    """
    idx, gt, generation_dir = args

    result_for_sample: Dict[str, Any] = {"ground_truth": gt}

    # ------------------------------------------------------------------
    # 1. Evaluate the original prediction
    # ------------------------------------------------------------------
    original_key = f"{idx:03d}_original"
    original_path = generation_dir / f"{original_key}.json"

    result_for_sample[original_key] = {}
    try:
        original_text = original_path.read_text(encoding="utf-8")
        try:
            # Validate JSON structure (content unchanged for distance calc)
            original_text = json.dumps(json.loads(original_text), ensure_ascii=False, indent=4)
            is_json_structured = True
        except json.JSONDecodeError:
            is_json_structured = False
        result_for_sample[original_key].update(
            {
                "is_json_structured": is_json_structured,
                "generated_text": original_text,
                "edit_distance_to_gt": edit_distance(original_text, gt),
            }
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Original generation file not found: {e.filename}")

    # ------------------------------------------------------------------
    # 2. Evaluate the (cut, extent) variants
    # ------------------------------------------------------------------
    cut = ["half_cut", "third_cut", "quarter_cut"]
    extent = ["0", "1", "2", "3"]

    for c, e in product(cut, extent):
        generation_key = f"{idx:03d}_{c}_{e}"
        generation_path = generation_dir / f"{generation_key}.json"
        result_for_sample[generation_key] = {}
        try:
            generated_text = generation_path.read_text(encoding="utf-8")
            try:
                generated_text = json.dumps(json.loads(generated_text), ensure_ascii=False, indent=4)
                is_json_structured = True
            except json.JSONDecodeError:
                is_json_structured = False

            result_for_sample[generation_key].update(
                {
                    "is_json_structured": is_json_structured,
                    "generated_text": generated_text,
                    "edit_distance_to_gt": edit_distance(generated_text, gt),
                    "edit_distance_to_original": edit_distance(
                        generated_text, result_for_sample[original_key]["generated_text"]
                    ),
                }
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Generation file not found: {e.filename}")

    return idx, result_for_sample

###############################################################
# Main evaluation routine ------------------------------------#
###############################################################

def make_generation_res_mp(
    generation_dir: Path, gts: List[str], save_fname: Path, num_workers: int = None, tqdm_position: int = 0
) -> List[Dict[str, Any]]:
    """Compute edit-distance based evaluation in parallel **with progress bar**.

    Parameters
    ----------
    generation_dir : Path
        Directory containing model generations as JSON files.
    gts : List[str]
        Ground-truth strings (already JSON-serialised, one per sample).
    save_fname : Path
        Destination .jsonl file where each line is a per-sample result.
    num_workers : int, optional
        Number of worker processes. Defaults to `os.cpu_count()`.
    tqdm_position : int, optional
        "position" argument for tqdm, used when rendering multiple bars.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered list of evaluation dictionaries (same order as *gts*).
    """

    num_workers = num_workers or cpu_count()

    # ==========================================================================
    # 1. Launch worker pool & progress bar
    # ==========================================================================
    intermediate: Dict[int, Dict[str, Any]] = {}
    with Pool(processes=num_workers) as pool, tqdm(
        total=len(gts),
        desc=f"{generation_dir.name}",
        dynamic_ncols=True,
        position=tqdm_position,
        leave=False,
    ) as pbar:
        iterable = ((i, gts[i], generation_dir) for i in range(len(gts)))
        for idx, res in pool.imap_unordered(_process_one, iterable, chunksize=16):
            intermediate[idx] = res
            pbar.update()

    # ==========================================================================
    # 2. Restore order & write JSONL in a single pass
    # ==========================================================================
    ordered_results = [intermediate[i] for i in range(len(gts))]

    save_fname.parent.mkdir(parents=True, exist_ok=True)
    with save_fname.open("w", encoding="utf-8") as f:
        for sample_res in ordered_results:
            f.write(json.dumps(sample_res, ensure_ascii=False) + "\n")

    return ordered_results

###############################################################
# Command-line interface -------------------------------------#
###############################################################

def _parse_generation_pairs(raw_pairs: List[str]) -> List[Tuple[Path, Path]]:
    """Convert CLI arg list of `dir:save_file` into Path tuples."""
    pairs = []
    for raw in raw_pairs:
        try:
            dir_part, file_part = raw.split(":", 1)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "Each --pairs argument must be of the form generation_dir:save_file.jsonl"
            ) from e
        pairs.append((Path(dir_part).expanduser(), Path(file_part).expanduser()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel generation evaluator with progress bars")

    parser.add_argument(
        "--pairs",
        required=True,
        metavar="DIR:FILE",
        nargs="+",
        help="One or more pairs of generation_dir:output_jsonl",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="CORD-V2 split to use for ground truths (default: test)",
    )

    args = parser.parse_args()

    # ---------------------------------------------------
    # 1. Load ground truths once
    # ---------------------------------------------------
    ds = load_dataset("naver-clova-ix/cord-v2", split=args.split)
    gts = [json.loads(sample["ground_truth"])["gt_parse"] for sample in ds]
    gts = [json.dumps(gt, ensure_ascii=False, indent=4) for gt in gts]

    # ---------------------------------------------------
    # 2. Evaluate each (generation_dir, save_file) pair
    # ---------------------------------------------------
    for pos, (generation_dir, save_file) in enumerate(_parse_generation_pairs(args.pairs)):
        tqdm.write(f"\n▶ Evaluating {generation_dir} → {save_file}")
        make_generation_res_mp(
            generation_dir=generation_dir,
            gts=gts,
            save_fname=save_file,
            num_workers=args.num_workers,
            tqdm_position=pos,
        )
        tqdm.write(f"✓ Saved results to {save_file}\n")

    tqdm.write("All evaluations completed.")


if __name__ == "__main__":
    main()
