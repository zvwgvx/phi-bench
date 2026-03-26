import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "./models/llama-3.2-1b"
DEFAULT_TRIPLETS_PATH = "./data/toy_triplets.json"
DEFAULT_OUTPUT_PATH = "./outputs/phi_proxy_results.json"
DEFAULT_ABLATION_OUTPUT_PATH = "./outputs/phi_proxy_ablation.json"
PROMPT_KEYS = ("original", "surface", "inverse")
PROMPT_PREFIX = "Solve the following problem. Answer with only a number.\nProblem: "

@dataclass
class PromptCache:
    prompt: str
    model_prompt: str
    input_ids: List[int]
    prompt_token_indices: List[int]
    hidden_states: List[np.ndarray]
    generated_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype runner for Phi-proxy and PhiStability on toy triplets."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--triplets-path", default=DEFAULT_TRIPLETS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--aggregation",
        choices=("token_pooled", "question_end_window"),
        default="token_pooled",
    )
    parser.add_argument(
        "--projector",
        choices=("random", "pca", "truncated_pca"),
        default="random",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("half", "even_odd", "random"),
        default=("half", "even_odd"),
    )
    parser.add_argument("--d-proj", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--ablation-d-proj",
        nargs="+",
        type=int,
        default=(8, 32, 64, 128, 256, 512),
    )
    parser.add_argument(
        "--ablation-projectors",
        nargs="+",
        choices=("random", "pca", "truncated_pca"),
        default=("random", "pca", "truncated_pca"),
    )
    parser.add_argument("--ablation-output-path", default=DEFAULT_ABLATION_OUTPUT_PATH)
    return parser.parse_args()


def load_triplets(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Triplets file must contain a non-empty list.")
    for item in data:
        if "id" not in item:
            raise ValueError("Each triplet must contain an 'id'.")
        for key in PROMPT_KEYS:
            if key not in item or not isinstance(item[key], str):
                raise ValueError(f"Triplet '{item.get('id', '<unknown>')}' is missing '{key}'.")
        answers = item.get("answers")
        if not isinstance(answers, dict):
            raise ValueError(f"Triplet '{item['id']}' must contain an 'answers' object.")
        for key in PROMPT_KEYS:
            if key not in answers or not isinstance(answers[key], str):
                raise ValueError(f"Triplet '{item['id']}' is missing answers['{key}'].")
    return data


def build_model_prompt(prompt: str) -> str:
    return f"{PROMPT_PREFIX}{prompt}\nAnswer:"


def normalize_token(token: str) -> str:
    token = token.replace("▁", "").replace("Ġ", "").strip()
    return token


def build_prompt_token_indices(
    input_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    tokenizer,
    prompt_start: int,
    prompt_end: int,
) -> List[int]:
    special_ids = set(tokenizer.all_special_ids)
    indices: List[int] = []
    for idx, (token_id, offsets) in enumerate(zip(input_ids, offset_mapping)):
        if token_id in special_ids:
            continue
        start, end = int(offsets[0]), int(offsets[1])
        if end <= start:
            continue
        if end <= prompt_start or start >= prompt_end:
            continue
        indices.append(idx)
    if not indices:
        raise ValueError("Failed to locate prompt token span inside model prompt.")
    return indices


def build_content_indices(prompt_cache: PromptCache, tokenizer) -> np.ndarray:
    indices: List[int] = []
    special_ids = set(tokenizer.all_special_ids)
    tokens = tokenizer.convert_ids_to_tokens(prompt_cache.input_ids)
    allowed = set(prompt_cache.prompt_token_indices)

    for idx, (token_id, token) in enumerate(zip(prompt_cache.input_ids, tokens)):
        if idx not in allowed:
            continue
        if token_id in special_ids:
            continue
        clean = normalize_token(token)
        if not clean:
            continue
        if re.fullmatch(r"[\W_]+", clean):
            continue
        indices.append(idx)

    if not indices:
        raise ValueError("No content tokens selected for prompt.")
    return np.asarray(indices, dtype=np.int64)


def select_indices(content_indices: np.ndarray, aggregation: str, window_size: int) -> np.ndarray:
    if aggregation == "token_pooled":
        return content_indices
    if aggregation == "question_end_window":
        return content_indices[-window_size:]
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def collect_prompt_cache(
    triplets: Sequence[dict],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
) -> Dict[str, Dict[str, PromptCache]]:
    cache: Dict[str, Dict[str, PromptCache]] = {}
    with torch.inference_mode():
        for triplet in triplets:
            triplet_cache: Dict[str, PromptCache] = {}
            for key in PROMPT_KEYS:
                prompt = triplet[key]
                model_prompt = build_model_prompt(prompt)
                encoded = tokenizer(
                    model_prompt,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                offset_mapping = encoded.pop("offset_mapping")[0].tolist()
                prompt_start = len(PROMPT_PREFIX)
                prompt_end = prompt_start + len(prompt)
                input_ids_cpu = encoded["input_ids"][0].detach().cpu().tolist()
                prompt_token_indices = build_prompt_token_indices(
                    input_ids=input_ids_cpu,
                    offset_mapping=offset_mapping,
                    tokenizer=tokenizer,
                    prompt_start=prompt_start,
                    prompt_end=prompt_end,
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
                generated_ids = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated_text = tokenizer.decode(
                    generated_ids[0][encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                hidden_states = [
                    layer.squeeze(0).detach().cpu().to(torch.float32).numpy()
                    for layer in outputs.hidden_states[1:]
                ]
                triplet_cache[key] = PromptCache(
                    prompt=prompt,
                    model_prompt=model_prompt,
                    input_ids=input_ids_cpu,
                    prompt_token_indices=prompt_token_indices,
                    hidden_states=hidden_states,
                    generated_text=generated_text,
                )
            cache[triplet["id"]] = triplet_cache
    return cache


class RandomProjector:
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((input_dim, output_dim))
        q, _ = np.linalg.qr(matrix)
        self.matrix = q[:, :output_dim]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values @ self.matrix


class PCAProjector:
    def __init__(self, matrix: np.ndarray, output_dim: int):
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.mean = matrix.mean(axis=0, keepdims=True)
        self.components = vh[:output_dim].T

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) @ self.components


class TruncatedPCAProjector:
    def __init__(self, matrix: np.ndarray, output_dim: int, drop_components: int):
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        start = min(drop_components, vh.shape[0] - output_dim)
        self.mean = matrix.mean(axis=0, keepdims=True)
        self.components = vh[start : start + output_dim].T

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) @ self.components


def gather_projection_matrix(
    cache: Dict[str, Dict[str, PromptCache]],
    tokenizer,
    aggregation: str,
    window_size: int,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for triplet_cache in cache.values():
        for prompt_cache in triplet_cache.values():
            content_indices = build_content_indices(prompt_cache, tokenizer)
            selected_indices = select_indices(content_indices, aggregation, window_size)
            for layer in prompt_cache.hidden_states:
                rows.append(layer[selected_indices])
    return np.concatenate(rows, axis=0)


def collect_sample_counts(
    cache: Dict[str, Dict[str, PromptCache]],
    tokenizer,
    aggregation: str,
    window_size: int,
) -> List[int]:
    counts: List[int] = []
    for triplet_cache in cache.values():
        for prompt_cache in triplet_cache.values():
            content_indices = build_content_indices(prompt_cache, tokenizer)
            selected_indices = select_indices(content_indices, aggregation, window_size)
            counts.append(int(selected_indices.shape[0]))
    return counts


def resolve_effective_d_proj(requested_d_proj: int, sample_counts: Sequence[int]) -> Dict[str, float]:
    if not sample_counts:
        raise ValueError("No sample counts available to resolve d_proj.")
    min_sample_count = int(min(sample_counts))
    max_candidate = max(2, (min_sample_count - 1) // 2)
    max_even_d_proj = max_candidate if max_candidate % 2 == 0 else max_candidate - 1
    max_even_d_proj = max(2, max_even_d_proj)
    effective_d_proj = min(requested_d_proj, max_even_d_proj)
    if effective_d_proj % 2 != 0:
        effective_d_proj -= 1
    effective_d_proj = max(2, effective_d_proj)
    return {
        "requested_d_proj": int(requested_d_proj),
        "effective_d_proj": int(effective_d_proj),
        "min_prompt_sample_count": min_sample_count,
        "max_prompt_sample_count": int(max(sample_counts)),
        "mean_prompt_sample_count": float(np.mean(sample_counts)),
    }


def create_projector(
    projector_name: str,
    projection_matrix: np.ndarray,
    output_dim: int,
    seed: int,
):
    input_dim = projection_matrix.shape[1]
    if output_dim <= 0 or output_dim > input_dim:
        raise ValueError("d_proj must be in [1, hidden_size].")
    if projector_name == "random":
        return RandomProjector(input_dim=input_dim, output_dim=output_dim, seed=seed)
    if projector_name == "pca":
        return PCAProjector(matrix=projection_matrix, output_dim=output_dim)
    if projector_name == "truncated_pca":
        drop_components = min(max(4, output_dim // 2), max(0, input_dim - output_dim))
        return TruncatedPCAProjector(
            matrix=projection_matrix,
            output_dim=output_dim,
            drop_components=drop_components,
        )
    raise ValueError(f"Unsupported projector: {projector_name}")


def build_partition_indices(d_proj: int, partition_name: str, seed: int) -> np.ndarray:
    if d_proj % 2 != 0:
        raise ValueError("d_proj must be even so that A/B partitions are balanced.")

    if partition_name == "half":
        return np.arange(d_proj)
    if partition_name == "even_odd":
        even = np.arange(0, d_proj, 2)
        odd = np.arange(1, d_proj, 2)
        return np.concatenate([even, odd])
    if partition_name == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(d_proj)
    raise ValueError(f"Unsupported partition: {partition_name}")


def reorder_and_split(values: np.ndarray, ordering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reordered = values[:, ordering]
    half = reordered.shape[1] // 2
    return reordered[:, :half], reordered[:, half:]


def regularized_covariance(values: np.ndarray, ridge: float) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(values.shape[0] - 1, 1)
    scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    scale = scale if math.isfinite(scale) and scale > 0 else 1.0
    return cov + ridge * scale * np.eye(cov.shape[0], dtype=np.float64)


def gaussian_mutual_information(x: np.ndarray, y: np.ndarray, ridge: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    cov_x = regularized_covariance(x, ridge)
    cov_y = regularized_covariance(y, ridge)
    cov_xy = regularized_covariance(np.concatenate([x, y], axis=1), ridge)

    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)
    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        raise ValueError("Covariance matrix is not positive definite after regularization.")
    return 0.5 * (logdet_x + logdet_y - logdet_xy)


def phi_proxy_gaussian(
    a_l: np.ndarray,
    b_l: np.ndarray,
    a_lp1: np.ndarray,
    b_lp1: np.ndarray,
    ridge: float,
) -> Dict[str, float]:
    joint_l = np.concatenate([a_l, b_l], axis=1)
    joint_lp1 = np.concatenate([a_lp1, b_lp1], axis=1)

    mi_joint = gaussian_mutual_information(joint_lp1, joint_l, ridge)
    mi_a = gaussian_mutual_information(a_lp1, a_l, ridge)
    mi_b = gaussian_mutual_information(b_lp1, b_l, ridge)
    signed = mi_joint - mi_a - mi_b
    return {
        "mi_joint": float(mi_joint),
        "mi_a": float(mi_a),
        "mi_b": float(mi_b),
        "phi_signed": float(signed),
        "phi_clipped": float(max(0.0, signed)),
    }


def build_projected_layers(
    prompt_cache: PromptCache,
    tokenizer,
    projector,
    aggregation: str,
    window_size: int,
) -> tuple[List[np.ndarray], int]:
    content_indices = build_content_indices(prompt_cache, tokenizer)
    selected_indices = select_indices(content_indices, aggregation, window_size)
    projected_layers: List[np.ndarray] = []
    for layer in prompt_cache.hidden_states:
        projected_layers.append(projector.transform(layer[selected_indices]))
    return projected_layers, int(selected_indices.shape[0])


def compute_prompt_profile(
    prompt_cache: PromptCache,
    tokenizer,
    projector,
    aggregation: str,
    window_size: int,
    partitions: Sequence[str],
    d_proj: int,
    seed: int,
    ridge: float,
) -> Dict[str, object]:
    projected_layers, sample_count = build_projected_layers(
        prompt_cache=prompt_cache,
        tokenizer=tokenizer,
        projector=projector,
        aggregation=aggregation,
        window_size=window_size,
    )
    if sample_count < 2:
        raise ValueError("At least two samples are required to estimate covariance.")

    partition_orders = {
        name: build_partition_indices(d_proj=d_proj, partition_name=name, seed=seed)
        for name in partitions
    }

    layer_metrics = []
    for layer_index in range(len(projected_layers) - 1):
        z_l = projected_layers[layer_index]
        z_lp1 = projected_layers[layer_index + 1]

        partition_results = {}
        for partition_name, ordering in partition_orders.items():
            a_l, b_l = reorder_and_split(z_l, ordering)
            a_lp1, b_lp1 = reorder_and_split(z_lp1, ordering)
            partition_results[partition_name] = phi_proxy_gaussian(
                a_l=a_l,
                b_l=b_l,
                a_lp1=a_lp1,
                b_lp1=b_lp1,
                ridge=ridge,
            )

        signed_values = [item["phi_signed"] for item in partition_results.values()]
        clipped_values = [item["phi_clipped"] for item in partition_results.values()]
        layer_metrics.append(
            {
                "layer_index": layer_index,
                "phi_signed_mean": float(np.mean(signed_values)),
                "phi_clipped_mean": float(np.mean(clipped_values)),
                "partitions": partition_results,
            }
        )

    return {
        "sample_count": sample_count,
        "layer_metrics": layer_metrics,
        "phi_signed_profile": [item["phi_signed_mean"] for item in layer_metrics],
        "phi_clipped_profile": [item["phi_clipped_mean"] for item in layer_metrics],
    }


def compute_phi_stability(
    phi_original: Sequence[float],
    phi_surface: Sequence[float],
    phi_inverse: Sequence[float],
    epsilon: float = 1e-8,
) -> Dict[str, object]:
    stack = np.asarray([phi_original, phi_surface, phi_inverse], dtype=np.float64)
    dispersion = np.std(stack, axis=0) / (np.mean(np.abs(stack), axis=0) + epsilon)
    stability = np.exp(-dispersion)
    return {
        "dispersion_by_layer": dispersion.tolist(),
        "stability_by_layer": stability.tolist(),
        "stability_global": float(np.mean(stability)),
    }


def compute_pair_stability(
    phi_left: Sequence[float],
    phi_right: Sequence[float],
    epsilon: float = 1e-8,
) -> Dict[str, object]:
    left = np.asarray(phi_left, dtype=np.float64)
    right = np.asarray(phi_right, dtype=np.float64)
    dispersion = np.abs(left - right) / ((np.abs(left) + np.abs(right)) / 2.0 + epsilon)
    stability = np.exp(-dispersion)
    return {
        "dispersion_by_layer": dispersion.tolist(),
        "stability_by_layer": stability.tolist(),
        "stability_global": float(np.mean(stability)),
    }


def average_stability_dicts(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not items:
        raise ValueError("Cannot average an empty list of stability dictionaries.")
    dispersion = np.mean(
        [np.asarray(item["dispersion_by_layer"], dtype=np.float64) for item in items],
        axis=0,
    )
    stability = np.mean(
        [np.asarray(item["stability_by_layer"], dtype=np.float64) for item in items],
        axis=0,
    )
    stability_global = float(np.mean([item["stability_global"] for item in items]))
    return {
        "dispersion_by_layer": dispersion.tolist(),
        "stability_by_layer": stability.tolist(),
        "stability_global": stability_global,
    }


def subtract_stability_dicts(
    actual: Dict[str, object],
    null: Dict[str, object],
) -> Dict[str, object]:
    actual_stability = np.asarray(actual["stability_by_layer"], dtype=np.float64)
    null_stability = np.asarray(null["stability_by_layer"], dtype=np.float64)
    delta = actual_stability - null_stability
    return {
        "delta_by_layer": delta.tolist(),
        "delta_global": float(np.mean(delta)),
    }


def extract_number_candidates(text: str) -> List[str]:
    return [match.replace(",", ".") for match in re.findall(r"-?\d+(?:[.,]\d+)?", text)]


def extract_preferred_number(text: str) -> str:
    normalized = text.replace(",", ".")
    keyword_patterns = (
        r"answer\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"final\s+answer\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        r"therefore\s*(?:the\s+answer\s*(?:is|=|:))?\s*(-?\d+(?:\.\d+)?)",
        r"so\s*(?:the\s+answer\s*(?:is|=|:))?\s*(-?\d+(?:\.\d+)?)",
    )
    for pattern in keyword_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    candidates = extract_number_candidates(normalized)
    if candidates:
        return candidates[-1]
    return ""


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numbers_match(predicted: str, gold: str, tolerance: float = 1e-6) -> bool:
    predicted_value = parse_number(predicted)
    gold_value = parse_number(gold)
    if predicted_value is None or gold_value is None:
        return False
    return abs(predicted_value - gold_value) <= tolerance * max(1.0, abs(gold_value))


def score_prediction(prediction: str, answer: str) -> Dict[str, object]:
    predicted_number = extract_preferred_number(prediction)
    gold_number = extract_preferred_number(answer)
    exact_match = prediction.strip() == answer.strip()
    numeric_match = numbers_match(predicted_number, gold_number)
    return {
        "prediction_text": prediction,
        "gold_text": answer,
        "prediction_number": predicted_number,
        "gold_number": gold_number,
        "exact_match": bool(exact_match),
        "numeric_match": bool(numeric_match),
        "is_correct": bool(exact_match or numeric_match),
    }


def prepare_experiment(
    args: argparse.Namespace,
) -> tuple[List[dict], object, Dict[str, Dict[str, PromptCache]], np.ndarray, Dict[str, float]]:
    triplets_path = Path(args.triplets_path)
    triplets = load_triplets(triplets_path)
    tokenizer, model = load_model(args.model_path, args.device)
    cache = collect_prompt_cache(
        triplets=triplets,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    del model

    projection_matrix = gather_projection_matrix(
        cache=cache,
        tokenizer=tokenizer,
        aggregation=args.aggregation,
        window_size=args.window_size,
    )
    sample_counts = collect_sample_counts(
        cache=cache,
        tokenizer=tokenizer,
        aggregation=args.aggregation,
        window_size=args.window_size,
    )
    d_proj_info = resolve_effective_d_proj(args.d_proj, sample_counts)
    return triplets, tokenizer, cache, projection_matrix, d_proj_info


def execute_experiment_with_prepared(
    args: argparse.Namespace,
    triplets: Sequence[dict],
    tokenizer,
    cache: Dict[str, Dict[str, PromptCache]],
    projection_matrix: np.ndarray,
    d_proj_info: Dict[str, float],
) -> dict:
    triplets_path = Path(args.triplets_path)
    effective_d_proj = int(d_proj_info["effective_d_proj"])
    projector = create_projector(
        projector_name=args.projector,
        projection_matrix=projection_matrix,
        output_dim=effective_d_proj,
        seed=args.random_seed,
    )

    results = []
    for triplet in triplets:
        triplet_cache = cache[triplet["id"]]
        prompt_results = {}
        scores = {}
        for prompt_key in PROMPT_KEYS:
            prompt_results[prompt_key] = compute_prompt_profile(
                prompt_cache=triplet_cache[prompt_key],
                tokenizer=tokenizer,
                projector=projector,
                aggregation=args.aggregation,
                window_size=args.window_size,
                partitions=args.partitions,
                d_proj=effective_d_proj,
                seed=args.random_seed,
                ridge=args.ridge,
            )
            scores[prompt_key] = score_prediction(
                prediction=triplet_cache[prompt_key].generated_text,
                answer=triplet["answers"][prompt_key],
            )

        phi_stability = compute_phi_stability(
            phi_original=prompt_results["original"]["phi_signed_profile"],
            phi_surface=prompt_results["surface"]["phi_signed_profile"],
            phi_inverse=prompt_results["inverse"]["phi_signed_profile"],
        )
        surface_stability = compute_pair_stability(
            phi_left=prompt_results["original"]["phi_signed_profile"],
            phi_right=prompt_results["surface"]["phi_signed_profile"],
        )
        inverse_stability = compute_pair_stability(
            phi_left=prompt_results["original"]["phi_signed_profile"],
            phi_right=prompt_results["inverse"]["phi_signed_profile"],
        )

        results.append(
            {
                "id": triplet["id"],
                "aggregation": args.aggregation,
                "projector": args.projector,
                "prompt_results": prompt_results,
                "scores": scores,
                "surface_stability": surface_stability,
                "inverse_stability": inverse_stability,
                "phi_stability": phi_stability,
            }
        )

    for index, result in enumerate(results):
        surface_null_candidates = []
        inverse_null_candidates = []
        for other_index, other in enumerate(results):
            if index == other_index:
                continue
            surface_null_candidates.append(
                compute_pair_stability(
                    phi_left=result["prompt_results"]["original"]["phi_signed_profile"],
                    phi_right=other["prompt_results"]["surface"]["phi_signed_profile"],
                )
            )
            inverse_null_candidates.append(
                compute_pair_stability(
                    phi_left=result["prompt_results"]["original"]["phi_signed_profile"],
                    phi_right=other["prompt_results"]["inverse"]["phi_signed_profile"],
                )
            )

        surface_null = average_stability_dicts(surface_null_candidates)
        inverse_null = average_stability_dicts(inverse_null_candidates)
        result["surface_null_stability"] = surface_null
        result["inverse_null_stability"] = inverse_null
        result["surface_delta_stability"] = subtract_stability_dicts(
            result["surface_stability"], surface_null
        )
        result["inverse_delta_stability"] = subtract_stability_dicts(
            result["inverse_stability"], inverse_null
        )

    accuracy_values = [
        int(item["scores"][prompt_key]["is_correct"])
        for item in results
        for prompt_key in PROMPT_KEYS
    ]
    summary = {
        "triplet_count": len(results),
        "overall_accuracy": float(np.mean(accuracy_values)) if accuracy_values else 0.0,
        "mean_phi_stability_global": float(
            np.mean([item["phi_stability"]["stability_global"] for item in results])
        ),
        "mean_surface_stability_global": float(
            np.mean([item["surface_stability"]["stability_global"] for item in results])
        ),
        "mean_inverse_stability_global": float(
            np.mean([item["inverse_stability"]["stability_global"] for item in results])
        ),
        "mean_surface_null_stability_global": float(
            np.mean([item["surface_null_stability"]["stability_global"] for item in results])
        ),
        "mean_inverse_null_stability_global": float(
            np.mean([item["inverse_null_stability"]["stability_global"] for item in results])
        ),
        "mean_surface_delta_stability_global": float(
            np.mean([item["surface_delta_stability"]["delta_global"] for item in results])
        ),
        "mean_inverse_delta_stability_global": float(
            np.mean([item["inverse_delta_stability"]["delta_global"] for item in results])
        ),
    }

    return {
        "config": {
            "model_path": args.model_path,
            "triplets_path": str(triplets_path),
            "aggregation": args.aggregation,
            "projector": args.projector,
            "partitions": list(args.partitions),
            "requested_d_proj": args.d_proj,
            "effective_d_proj": effective_d_proj,
            "min_prompt_sample_count": d_proj_info["min_prompt_sample_count"],
            "max_prompt_sample_count": d_proj_info["max_prompt_sample_count"],
            "mean_prompt_sample_count": d_proj_info["mean_prompt_sample_count"],
            "window_size": args.window_size,
            "ridge": args.ridge,
            "random_seed": args.random_seed,
            "device": args.device,
        },
        "summary": summary,
        "results": results,
    }


def execute_experiment(args: argparse.Namespace) -> dict:
    triplets, tokenizer, cache, projection_matrix, d_proj_info = prepare_experiment(args)
    return execute_experiment_with_prepared(
        args=args,
        triplets=triplets,
        tokenizer=tokenizer,
        cache=cache,
        projection_matrix=projection_matrix,
        d_proj_info=d_proj_info,
    )


def run_experiment(args: argparse.Namespace) -> tuple[dict, Path]:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = execute_experiment(args)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output_path


def run_ablation(args: argparse.Namespace) -> tuple[dict, Path]:
    output_path = Path(args.ablation_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_d_proj = args.d_proj
    original_projector = args.projector
    original_quiet = args.quiet

    triplets, tokenizer, cache, projection_matrix, d_proj_info = prepare_experiment(args)

    records = []
    for d_proj in args.ablation_d_proj:
        for projector in args.ablation_projectors:
            args.d_proj = d_proj
            args.projector = projector
            args.quiet = True
            payload = execute_experiment_with_prepared(
                args=args,
                triplets=triplets,
                tokenizer=tokenizer,
                cache=cache,
                projection_matrix=projection_matrix,
                d_proj_info=d_proj_info,
            )
            summary = payload["summary"]
            records.append(
                {
                    "requested_d_proj": d_proj,
                    "effective_d_proj": payload["config"]["effective_d_proj"],
                    "projector": projector,
                    "overall_accuracy": summary["overall_accuracy"],
                    "mean_phi_stability_global": summary["mean_phi_stability_global"],
                    "mean_surface_stability_global": summary["mean_surface_stability_global"],
                    "mean_inverse_stability_global": summary["mean_inverse_stability_global"],
                    "mean_surface_null_stability_global": summary[
                        "mean_surface_null_stability_global"
                    ],
                    "mean_inverse_null_stability_global": summary[
                        "mean_inverse_null_stability_global"
                    ],
                    "mean_surface_delta_stability_global": summary[
                        "mean_surface_delta_stability_global"
                    ],
                    "mean_inverse_delta_stability_global": summary[
                        "mean_inverse_delta_stability_global"
                    ],
                }
            )

    args.d_proj = original_d_proj
    args.projector = original_projector
    args.quiet = original_quiet

    ablation_payload = {
        "config": {
            "model_path": args.model_path,
            "triplets_path": args.triplets_path,
            "aggregation": args.aggregation,
            "partitions": list(args.partitions),
            "window_size": args.window_size,
            "ridge": args.ridge,
            "random_seed": args.random_seed,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
        },
        "records": records,
    }
    output_path.write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
    return ablation_payload, output_path


def print_summary(payload: dict, output_path: Path) -> None:
    print(f"Results saved to: {output_path}")
    print(
        "Summary:"
        f" triplets={payload['summary']['triplet_count']},"
        f" overall_accuracy={payload['summary']['overall_accuracy']:.4f},"
        f" requested_d_proj={payload['config']['requested_d_proj']},"
        f" effective_d_proj={payload['config']['effective_d_proj']},"
        f" mean_phi_stability_global={payload['summary']['mean_phi_stability_global']:.4f},"
        f" mean_surface_stability_global={payload['summary']['mean_surface_stability_global']:.4f},"
        f" mean_inverse_stability_global={payload['summary']['mean_inverse_stability_global']:.4f},"
        f" mean_surface_null_stability_global={payload['summary']['mean_surface_null_stability_global']:.4f},"
        f" mean_inverse_null_stability_global={payload['summary']['mean_inverse_null_stability_global']:.4f},"
        f" mean_surface_delta_stability_global={payload['summary']['mean_surface_delta_stability_global']:.4f},"
        f" mean_inverse_delta_stability_global={payload['summary']['mean_inverse_delta_stability_global']:.4f}"
    )
    for item in payload["results"]:
        stability = item["phi_stability"]["stability_global"]
        surface_stability = item["surface_stability"]["stability_global"]
        inverse_stability = item["inverse_stability"]["stability_global"]
        surface_null = item["surface_null_stability"]["stability_global"]
        inverse_null = item["inverse_null_stability"]["stability_global"]
        surface_delta = item["surface_delta_stability"]["delta_global"]
        inverse_delta = item["inverse_delta_stability"]["delta_global"]
        score_summary = ", ".join(
            f"{key}={'1' if item['scores'][key]['is_correct'] else '0'}"
            for key in PROMPT_KEYS
        )
        print(
            f"  - {item['id']}:"
            f" phi_stability_global={stability:.4f},"
            f" surface_stability_global={surface_stability:.4f},"
            f" inverse_stability_global={inverse_stability:.4f},"
            f" surface_null={surface_null:.4f},"
            f" inverse_null={inverse_null:.4f},"
            f" surface_delta={surface_delta:.4f},"
            f" inverse_delta={inverse_delta:.4f},"
            f" scores[{score_summary}]"
        )


def print_ablation_summary(payload: dict, output_path: Path) -> None:
    print(f"Ablation results saved to: {output_path}")
    print(
        "req_d | eff_d | projector | accuracy | "
        "surface_stability | surface_null | surface_delta | "
        "inverse_stability | inverse_null | inverse_delta | phi_stability"
    )
    for record in payload["records"]:
        print(
            f"{record['requested_d_proj']:>5} | "
            f"{record['effective_d_proj']:>5} | "
            f"{record['projector']:<13} | "
            f"{record['overall_accuracy']:.4f} | "
            f"{record['mean_surface_stability_global']:.4f} | "
            f"{record['mean_surface_null_stability_global']:.4f} | "
            f"{record['mean_surface_delta_stability_global']:.4f} | "
            f"{record['mean_inverse_stability_global']:.4f} | "
            f"{record['mean_inverse_null_stability_global']:.4f} | "
            f"{record['mean_inverse_delta_stability_global']:.4f} | "
            f"{record['mean_phi_stability_global']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.ablation:
        payload, output_path = run_ablation(args)
        if not args.quiet:
            print_ablation_summary(payload, output_path)
    else:
        payload, output_path = run_experiment(args)
        if not args.quiet:
            print_summary(payload, output_path)


if __name__ == "__main__":
    main()
