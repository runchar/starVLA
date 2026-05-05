#!/usr/bin/env python3
"""Convert VSI-590K JSONL into the StarVLA VLM dataloader convention."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Path to VSI-590K vsi_590k.jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Path to write converted StarVLA JSONL")
    parser.add_argument("--media-root", default="", help="Optional root used with --require-media-exists")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Keep only media paths whose first path component matches this prefix. Can be repeated.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after writing this many samples; 0 means all")
    parser.add_argument(
        "--require-media-exists",
        action="store_true",
        help="Skip samples whose media file does not exist under --media-root",
    )
    return parser.parse_args()


def replace_first_visual_token(conversations, old_token, new_token):
    conversations = json.loads(json.dumps(conversations, ensure_ascii=False))
    for turn in conversations:
        if turn.get("from") == "human" and isinstance(turn.get("value"), str):
            turn["value"] = turn["value"].replace(old_token, new_token, 1)
            break
    return conversations


def convert_sample(sample):
    if "image" in sample:
        converted = {k: v for k, v in sample.items() if k not in {"image", "video"}}
        converted["images"] = sample["image"]
        converted["conversations"] = sample["conversations"]
        return converted, "image", sample["image"]
    if "video" in sample:
        converted = {k: v for k, v in sample.items() if k not in {"image", "video"}}
        converted["videos"] = sample["video"]
        converted["conversations"] = replace_first_visual_token(sample["conversations"], "<image>", "<video>")
        return converted, "video", sample["video"]
    return None, "missing_media", ""


def ordered_sample(sample):
    ordered = {}
    for key in ("images", "videos", "question_type", "id", "conversations"):
        if key in sample:
            ordered[key] = sample[key]
    for key, value in sample.items():
        ordered.setdefault(key, value)
    return ordered


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    media_root = Path(args.media_root) if args.media_root else None
    keep_prefixes = set(args.prefix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    written = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            counts["read"] += 1
            sample = json.loads(line)
            converted, media_type, media_path = convert_sample(sample)
            counts[media_type] += 1
            if converted is None:
                counts["skipped_missing_media"] += 1
                continue

            prefix = media_path.split("/", 1)[0]
            if keep_prefixes and prefix not in keep_prefixes:
                counts["skipped_prefix"] += 1
                continue

            if args.require_media_exists:
                if media_root is None:
                    raise ValueError("--require-media-exists requires --media-root")
                if not (media_root / media_path).exists():
                    counts["skipped_missing_file"] += 1
                    continue

            dst.write(json.dumps(ordered_sample(converted), ensure_ascii=False) + "\n")
            written += 1
            counts[f"written_{media_type}"] += 1

            if args.max_samples and written >= args.max_samples:
                break

    counts["written"] = written
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
