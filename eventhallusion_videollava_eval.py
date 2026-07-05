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
from tqdm.auto import tqdm

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


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
        "entire_questions.json",
        "eventhallusion_entire_rare.json",
    ],
    "mix": [
        "mix_common_rare.json",
        "mix.json",
        "mix_questions.json",
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
        path_str = str(p)
        rel_str = str(p.relative_to(root))
        # Store a few aliases so we can resolve different archive layouts.
        for key in {
            p.stem,
            p.name,
            p.with_suffix("").name,
            rel_str,
            Path(rel_str).with_suffix("").as_posix(),
        }:
            index[key] = path_str
    return index


def find_video(video_id: str, video_index: Dict[str, str], split: Optional[str] = None) -> str:
    if video_id in video_index:
        return video_index[video_id]

    if "_" in video_id:
        left, right = video_id.split("_", 1)
        rev = f"{right}_{left}"
        if rev in video_index:
            return video_index[rev]

        if split == "mix":
            interleave_id = f"interleave_{right}"
            if interleave_id in video_index:
                return video_index[interleave_id]

    # Some releases append clip-specific suffixes, e.g. mix_099_clip_1.mp4.
    # Fall back to a prefix match so the question id still resolves.
    candidates = sorted(
        (k for k in video_index if k.startswith(video_id + "_") or k.startswith(video_id + "-")),
        key=len,
    )
    if candidates:
        return video_index[candidates[0]]

    if split == "mix":
        mix_suffix = video_id.split("_", 1)[-1] if "_" in video_id else video_id
        mix_candidates = sorted(
            (
                k
                for k in video_index
                if k.startswith("interleave_" + mix_suffix)
                or k.endswith("/interleave_" + mix_suffix)
                or k.endswith("/" + mix_suffix)
            ),
            key=len,
        )
        if mix_candidates:
            return video_index[mix_candidates[0]]

    # Last resort: try matching on the split prefix and numeric suffix.
    if "_" in video_id:
        prefix, suffix = video_id.split("_", 1)
        loose_candidates = sorted(
            (
                k
                for k in video_index
                if (prefix in k and suffix in k)
                or k.endswith("/" + suffix + ".mp4")
                or k.endswith("/" + suffix)
                or k.startswith(suffix)
            ),
            key=len,
        )
        if loose_candidates:
            return video_index[loose_candidates[0]]

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


def gaussian_sigma_ratio(sigma: float, pixel_max: float = 255.0) -> float:
    """Approximate noise scale relative to the 8-bit pixel range."""
    return float(sigma) / float(pixel_max)


def summarize_spatial_gaussian(sigma: float) -> str:
    ratio = gaussian_sigma_ratio(sigma)
    return f"sigma={sigma:.1f}, approx_noise_ratio={ratio:.3f} of pixel range"


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


def build_prompt(question: str, context_prefix: str = "") -> str:
    prefix = context_prefix.strip()
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append("Answer the question using ONLY one word: Yes or No.")
    parts.append("")
    parts.append(f"Question:\n{question}")
    return "\n".join(parts) + "\n"


def build_language_bias_prompt(question: str, context_prefix: str) -> str:
    """Create a prompt variant that injects context bias without touching the video."""
    return build_prompt(question, context_prefix=context_prefix)


def infer_frames(model, processor, frames: np.ndarray, question: str, max_new_tokens: int = 32) -> str:
    return infer_frames_prompt(
        model=model,
        processor=processor,
        frames=frames,
        prompt=build_prompt(question),
        max_new_tokens=max_new_tokens,
    )


def infer_frames_prompt(
    model,
    processor,
    frames: np.ndarray,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    import torch

    full_prompt = f"USER: <video>\n{prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, videos=frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    answer = processor.batch_decode(output, skip_special_tokens=True)[0]
    if "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[-1]
    return normalize_yes_no(answer)


def make_background_removed_video(
    frames: np.ndarray,
    threshold: int = 25,
    apply_morphology: bool = True,
) -> np.ndarray:
    """Approximate background removal using a median background estimate.

    This is a lightweight proxy for semantic segmentation:
    - estimate background as the per-pixel median across frames
    - keep only pixels that differ sufficiently from that background
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for background removal")

    background = np.median(frames, axis=0).astype(np.uint8)
    outputs = []
    kernel = np.ones((3, 3), np.uint8)
    for frame in frames:
        diff = np.mean(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2)
        mask = (diff > threshold).astype(np.uint8) * 255
        if apply_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        outputs.append(fg)

    return np.stack(outputs, axis=0)


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
    show_progress: bool = True,
    condition_name: str = "normal",
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
    print(f"[{condition_name}] samples: {len(samples)}")
    sample_iter = tqdm(samples, desc=f"{condition_name} videos", leave=True) if show_progress else samples
    for sample in sample_iter:
        video_path = find_video(sample.video_id, video_index, split=sample.split)
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

        qa_iter = tqdm(sample.questions, desc=f"{condition_name}:{sample.video_id}", leave=False) if show_progress and len(sample.questions) > 1 else sample.questions
        for qa in qa_iter:
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
        if show_progress and hasattr(sample_iter, "set_postfix"):
            sample_iter.set_postfix(split=sample.split, video=sample.video_id)

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
    show_progress: bool = True,
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
        show_progress=show_progress,
        condition_name="normal",
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
        show_progress=show_progress,
        condition_name="spatial_gaussian",
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


def run_language_bias_eval(
    model,
    processor,
    questions_root: str,
    video_root: str,
    n_total_videos: int = 100,
    n_frames: int = 4,
    seed: int = 42,
    context_prefixes: Optional[List[str]] = None,
    split: str = "misleading",
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Measure language bias by keeping the video fixed and swapping prompt context.

    Returns a dataframe with baseline/variant predictions and consistency scores.
    """
    if context_prefixes is None:
        context_prefixes = [
            "At the beach,",
            "At school,",
            "In a kitchen,",
            "In an office,",
        ]

    splits = load_eventhallusion_jsons(questions_root)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}")

    target = {k: 0 for k in SPLIT_CANDIDATES}
    target[split] = n_total_videos
    samples = sample_eventhallusion_videos(
        splits,
        n_total_videos=n_total_videos,
        seed=seed,
        per_split=target,
    )
    video_index = build_video_index(video_root)

    rows = []
    iterator = tqdm(samples, desc=f"language_bias[{split}]", leave=True) if show_progress else samples
    for sample in iterator:
        video_path = find_video(sample.video_id, video_index, split=sample.split)
        frames = load_video(video_path, n_frames=n_frames)
        qa = sample.questions[0]
        question = qa.get("question") or qa.get("Question") or qa.get("q") or ""
        gt = normalize_yes_no(qa.get("answer") or qa.get("Answer") or qa.get("gt") or "")

        baseline_prompt = build_prompt(question)
        baseline_pred = infer_frames_prompt(model, processor, frames, baseline_prompt)

        record = {
            "split": sample.split,
            "video_id": sample.video_id,
            "question": question,
            "gt": gt,
            "baseline_pred": baseline_pred,
            "baseline_correct": baseline_pred == gt,
            "video_path": video_path,
        }

        for prefix in context_prefixes:
            prompt = build_language_bias_prompt(question, prefix)
            variant_key = prefix.lower().replace(",", "").replace(" ", "_")
            pred = infer_frames_prompt(model, processor, frames, prompt)
            record[f"pred_{variant_key}"] = pred
            record[f"correct_{variant_key}"] = pred == gt
            record[f"same_as_baseline_{variant_key}"] = pred == baseline_pred

        rows.append(record)

    df = pd.DataFrame(rows)
    metrics: Dict[str, float] = {
        "acc_baseline": float(df["baseline_correct"].mean()),
    }
    for prefix in context_prefixes:
        variant_key = prefix.lower().replace(",", "").replace(" ", "_")
        metrics[f"acc_{variant_key}"] = float(df[f"correct_{variant_key}"].mean())
        metrics[f"same_{variant_key}"] = float(df[f"same_as_baseline_{variant_key}"].mean())

    return df, metrics


def run_context_bias_eval(
    model,
    processor,
    questions_root: str,
    video_root: str,
    n_total_videos: int = 100,
    n_frames: int = 4,
    seed: int = 42,
    split: str = "misleading",
    show_progress: bool = True,
    use_background_removal: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Measure context bias by comparing normal videos vs background-removed videos.
    """
    splits = load_eventhallusion_jsons(questions_root)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}")

    target = {k: 0 for k in SPLIT_CANDIDATES}
    target[split] = n_total_videos
    samples = sample_eventhallusion_videos(
        splits,
        n_total_videos=n_total_videos,
        seed=seed,
        per_split=target,
    )
    video_index = build_video_index(video_root)

    rows = []
    iterator = tqdm(samples, desc=f"context_bias[{split}]", leave=True) if show_progress else samples
    for sample in iterator:
        video_path = find_video(sample.video_id, video_index, split=sample.split)
        frames = load_video(video_path, n_frames=n_frames)
        qa = sample.questions[0]
        question = qa.get("question") or qa.get("Question") or qa.get("q") or ""
        gt = normalize_yes_no(qa.get("answer") or qa.get("Answer") or qa.get("gt") or "")

        base_pred = infer_frames_prompt(model, processor, frames, build_prompt(question))
        if use_background_removal:
            fg_frames = make_background_removed_video(frames)
            fg_pred = infer_frames_prompt(model, processor, fg_frames, build_prompt(question))
        else:
            fg_pred = base_pred

        rows.append(
            {
                "split": sample.split,
                "video_id": sample.video_id,
                "question": question,
                "gt": gt,
                "baseline_pred": base_pred,
                "baseline_correct": base_pred == gt,
                "foreground_pred": fg_pred,
                "foreground_correct": fg_pred == gt,
                "same_as_baseline": fg_pred == base_pred,
                "video_path": video_path,
            }
        )

    df = pd.DataFrame(rows)
    metrics = {
        "acc_baseline": float(df["baseline_correct"].mean()),
        "acc_foreground": float(df["foreground_correct"].mean()),
        "same_prediction_rate": float(df["same_as_baseline"].mean()),
    }
    return df, metrics


__all__ = [
    "EHSample",
    "build_video_index",
    "find_video",
    "load_video",
    "add_gaussian_noise",
    "make_spatial_negative",
    "gaussian_sigma_ratio",
    "summarize_spatial_gaussian",
    "build_prompt",
    "build_language_bias_prompt",
    "infer_frames",
    "infer_frames_prompt",
    "make_background_removed_video",
    "load_eventhallusion_jsons",
    "sample_eventhallusion_videos",
    "run_eventhallusion_eval",
    "run_language_bias_eval",
    "run_context_bias_eval",
    "save_results",
    "compare_conditions",
]
