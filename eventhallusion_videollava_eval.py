"""EventHallusion + VideoLLaVA evaluation helpers.

This module is meant to be imported from the existing VideoLLaVA notebook.
It assumes you already have:
  - `model`
  - `processor`
  - a `load_video(path, n_frames=8)` function, or you can use the one below.

The goal is to compare:
  1. Normal video input
  2. Spatial Gaussian negative input

This is useful for testing whether spatial negatives reduce language/context
shortcut behavior on EventHallusion, especially on:
  - misleading questions
  - entire rare events
  - mix common-rare events
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None


SPLIT_CANDIDATES = {
    "misleading": [
        "misleading.json",
        "misleading_questions.json",
        "eventhallusion_misleading.json",
    ],
    "entire": [
        "entire_rare.json",
        "entire_rare_events.json",
        "entire.json",
        "eventhallusion_entire_rare.json",
    ],
    "mix": [
        "mix_common_rare.json",
        "mix.json",
        "eventhallusion_mix_common_rare.json",
    ],
}


@dataclass
class EHSample:
    split: str
    video_id: str
    questions: List[Dict]
    meta: Dict


def build_video_index(video_root: str) -> Dict[str, str]:
    root = Path(video_root)
    index: Dict[str, str] = {}
    for p in root.rglob("*.mp4"):
        index[p.stem] = str(p)
    return index


def find_video(video_id: str, video_index: Dict[str, str]) -> str:
    if video_id in video_index:
        return video_index[video_id]

    if "_" in video_id:
        left, right = video_id.split("_", 1)
        rev = f"{right}_{left}"
        if rev in video_index:
            return video_index[rev]

    raise FileNotFoundError(video_id)


def load_video(path: str, n_frames: int = 8) -> np.ndarray:
    if VideoReader is None:
        raise ImportError("decord is not available")

    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
    return vr.get_batch(idx).asnumpy()


def add_gaussian_noise(frame: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_spatial_negative(frames: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    return np.stack([add_gaussian_noise(f, sigma=sigma) for f in frames], axis=0)


def normalize_yes_no(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    up = text.upper()
    if up.startswith("YES"):
        return "Yes"
    if up.startswith("NO"):
        return "No"
    return text


def build_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY one word: Yes or No.\n\n"
        f"Question:\n{question}\n"
    )


def infer_frames(model, processor, frames: np.ndarray, question: str, max_new_tokens: int = 32) -> str:
    import torch

    prompt = f"USER: <video>\n{build_prompt(question)}\nASSISTANT:"
    inputs = processor(text=prompt, videos=frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )

    answer = processor.batch_decode(output, skip_special_tokens=True)[0]
    if "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[-1]
    return normalize_yes_no(answer)


def load_eventhallusion_jsons(questions_root: str) -> Dict[str, list]:
    root = Path(questions_root)
    out: Dict[str, list] = {}

    for split, candidates in SPLIT_CANDIDATES.items():
        found = None
        for name in candidates:
            p = root / name
            if p.exists():
                found = p
                break

        if found is None:
            raise FileNotFoundError(
                f"Could not find a JSON for split='{split}' under {questions_root}. "
                f"Tried: {candidates}"
            )

        with open(found, "r", encoding="utf-8") as f:
            out[split] = json.load(f)

    return out


def _iter_video_entries(split: str, data: list) -> Iterable[EHSample]:
    """
    Official EventHallusion question files are list[dict], one dict per video.
    Each item contains:
      - id
      - category
      - length
      - event_info
      - questions: [{"question": ..., "answer": ...}, ...]
    """
    if not isinstance(data, list):
        raise TypeError(f"Expected list for split='{split}', got {type(data).__name__}")

    for row in data:
        video_id = row.get("id") or row.get("video_id") or row.get("video")
        questions = row.get("questions") or []
        if video_id and isinstance(questions, list):
            yield EHSample(
                split=split,
                video_id=str(video_id),
                questions=questions,
                meta=row,
            )


def sample_eventhallusion_videos(
    splits: Dict[str, list],
    n_total_videos: int = 200,
    seed: int = 42,
    per_split: Optional[Dict[str, int]] = None,
) -> List[EHSample]:
    rng = random.Random(seed)
    per_split_entries: Dict[str, List[EHSample]] = {}

    for split, data in splits.items():
        per_split_entries[split] = list(_iter_video_entries(split, data))

    if not any(per_split_entries.values()):
        raise ValueError("No video entries were found in the provided EventHallusion JSONs.")

    if per_split is None:
        per_split = {k: n_total_videos // 3 for k in SPLIT_CANDIDATES.keys()}
        remainder = n_total_videos - sum(per_split.values())
        for split in list(SPLIT_CANDIDATES.keys())[:remainder]:
            per_split[split] += 1

    selected: List[EHSample] = []
    for split, k in per_split.items():
        pool = per_split_entries.get(split, [])
        if not pool:
            continue
        selected.extend(rng.sample(pool, min(k, len(pool))))

    rng.shuffle(selected)
    return selected


def run_eventhallusion_eval(
    model,
    processor,
    questions_root: str,
    video_root: str,
    n_total_videos: int = 200,
    n_frames: int = 8,
    sigma: float = 25.0,
    seed: int = 42,
    use_spatial_negative: bool = False,
    per_split: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], dict]:
    """
    Returns:
      df: row-level predictions
      metrics: accuracy by split + overall
    """
    splits = load_eventhallusion_jsons(questions_root)
    samples = sample_eventhallusion_videos(
        splits,
        n_total_videos=n_total_videos,
        seed=seed,
        per_split=per_split,
    )
    video_index = build_video_index(video_root)

    rows = []
    predictions: Dict[str, Dict[str, Dict]] = {}
    for sample in samples:
        video_path = find_video(sample.video_id, video_index)
        frames = load_video(video_path, n_frames=n_frames)
        if use_spatial_negative:
            frames = make_spatial_negative(frames, sigma=sigma)

        pred_video = {
            "id": sample.video_id,
            "qa": [],
        }
        if sample.split != "misleading":
            pred_video["desc"] = ""
            pred_video["judgement"] = ""

        for qa in sample.questions:
            question = qa.get("question") or qa.get("Question") or qa.get("q") or ""
            gt = normalize_yes_no(qa.get("answer") or qa.get("Answer") or qa.get("gt") or "")
            pred = infer_frames(model, processor, frames, question)

            pred_video["qa"].append(
                {
                    "question": question,
                    "answer": gt,
                    "prediction": pred,
                }
            )
            rows.append(
                {
                    "split": sample.split,
                    "video_id": sample.video_id,
                    "question": question,
                    "gt": gt,
                    "pred": pred,
                    "correct": gt == pred,
                    "condition": "spatial_gaussian" if use_spatial_negative else "normal",
                    "sigma": sigma if use_spatial_negative else None,
                    "n_frames": n_frames,
                    "video_path": video_path,
                }
            )

        predictions.setdefault(sample.split, {})[sample.video_id] = pred_video

    df = pd.DataFrame(rows)
    metrics = {"overall": float(df["correct"].mean())}
    for split in sorted(df["split"].unique()):
        metrics[f"acc_{split}"] = float(df.loc[df["split"] == split, "correct"].mean())
    return df, metrics, predictions


def save_results(df: pd.DataFrame, out_dir: str, prefix: str) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{prefix}.csv"
    json_path = out / f"{prefix}.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    return {"csv": str(csv_path), "json": str(json_path)}


def compare_conditions(
    model,
    processor,
    questions_root: str,
    video_root: str,
    out_dir: str = "./eventhallusion_results",
    n_total_videos: int = 200,
    n_frames: int = 8,
    sigma: float = 25.0,
    seed: int = 42,
    per_split: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that runs both:
      - normal
      - spatial_gaussian
    and merges the results into a single summary table.
    """
    normal_df, normal_metrics, normal_pred = run_eventhallusion_eval(
        model=model,
        processor=processor,
        questions_root=questions_root,
        video_root=video_root,
        n_total_videos=n_total_videos,
        n_frames=n_frames,
        sigma=sigma,
        seed=seed,
        use_spatial_negative=False,
        per_split=per_split,
    )
    neg_df, neg_metrics, neg_pred = run_eventhallusion_eval(
        model=model,
        processor=processor,
        questions_root=questions_root,
        video_root=video_root,
        n_total_videos=n_total_videos,
        n_frames=n_frames,
        sigma=sigma,
        seed=seed,
        use_spatial_negative=True,
        per_split=per_split,
    )

    normal_paths = save_results(normal_df, out_dir, "eventhallusion_normal")
    neg_paths = save_results(neg_df, out_dir, "eventhallusion_spatial_gaussian")

    with open(Path(out_dir) / "eventhallusion_normal_predictions.json", "w", encoding="utf-8") as f:
        json.dump(normal_pred, f, ensure_ascii=False, indent=2)
    with open(Path(out_dir) / "eventhallusion_spatial_gaussian_predictions.json", "w", encoding="utf-8") as f:
        json.dump(neg_pred, f, ensure_ascii=False, indent=2)

    summary = pd.DataFrame(
        [
            {"condition": "normal", **normal_metrics},
            {"condition": "spatial_gaussian", **neg_metrics},
        ]
    )

    summary_path = Path(out_dir) / "eventhallusion_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(normal_paths)
    print(neg_paths)
    print(str(summary_path))
    print(summary.to_string(index=False))

    return summary


__all__ = [
    "EHSample",
    "build_video_index",
    "find_video",
    "load_video",
    "add_gaussian_noise",
    "make_spatial_negative",
    "build_prompt",
    "infer_frames",
    "load_eventhallusion_jsons",
    "sample_eventhallusion_videos",
    "run_eventhallusion_eval",
    "save_results",
    "compare_conditions",
]
