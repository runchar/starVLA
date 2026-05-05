import re

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

json_root = f"./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons"
image_root = f"./playground/Datasets/LLaVA-OneVision-COCO/images"
vsi590k_root = "./playground/Datasets/VSI-590K"

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

VSI590K_RAW_SMOKE = {
    "annotation_path": f"{vsi590k_root}/annotations/vsi590k_raw_smoke.jsonl",
    "data_path": f"{vsi590k_root}/",
}

VSI590K_STARVLA_SMOKE = {
    "annotation_path": f"{vsi590k_root}/annotations/vsi590k_starvla_smoke.jsonl",
    "data_path": f"{vsi590k_root}/",
}

VSI590K_STARVLA_PARTIAL_DOWNLOADED = {
    "annotation_path": f"{vsi590k_root}/annotations/vsi590k_starvla_smoke.jsonl",
    "data_path": f"{vsi590k_root}/",
}

data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO,
    "vsi590k_raw_smoke": VSI590K_RAW_SMOKE,
    "vsi590k_starvla_smoke": VSI590K_STARVLA_SMOKE,
    "vsi590k_starvla_partial_downloaded": VSI590K_STARVLA_PARTIAL_DOWNLOADED,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    print(data_list)
    
