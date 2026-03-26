import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from run_phi_proxy import (
    build_partition_indices,
    build_projected_layers,
    create_projector,
    phi_proxy_gaussian,
    prepare_experiment,
    reorder_and_split,
)

DEFAULT_MODEL_PATH = "./models/llama-3.2-1b"
DEFAULT_TRIPLETS_PATH = "./data/toy_triplets.json"
DEFAULT_OUTPUT_PATH = "./outputs/partition_robustness.json"

PROMPT_KEYS = ("original", "surface", "inverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical test for partition robustness of Phi-proxy."
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
    parser.add_argument("--d-proj", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-random-partitions", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_partition_specs(d_proj: int, base_seed: int, num_random_partitions: int) -> List[dict]:
    specs = [
        {
            "name": "half",
            "family": "deterministic",
            "seed": None,
            "ordering": build_partition_indices(d_proj, "half", base_seed),
        },
        {
            "name": "even_odd",
            "family": "deterministic",
            "seed": None,
            "ordering": build_partition_indices(d_proj, "even_odd", base_seed),
        },
    ]
    for offset in range(num_random_partitions):
        seed = base_seed + offset
        specs.append(
            {
                "name": f"random_{seed}",
                "family": "random",
                "seed": seed,
                "ordering": build_partition_indices(d_proj, "random", seed),
            }
        )
    return specs


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denom <= 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denom)


def mean_pairwise_cosine(profiles: Sequence[Sequence[float]]) -> float:
    if len(profiles) < 2:
        return 1.0
    scores = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            scores.append(cosine_similarity(profiles[i], profiles[j]))
    return float(np.mean(scores)) if scores else 1.0


def compute_partition_profiles(
    projected_layers: Sequence[np.ndarray],
    partition_specs: Sequence[dict],
    ridge: float,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for spec in partition_specs:
        profile = []
        for layer_index in range(len(projected_layers) - 1):
            z_l = projected_layers[layer_index]
            z_lp1 = projected_layers[layer_index + 1]
            a_l, b_l = reorder_and_split(z_l, spec["ordering"])
            a_lp1, b_lp1 = reorder_and_split(z_lp1, spec["ordering"])
            phi_values = phi_proxy_gaussian(
                a_l=a_l,
                b_l=b_l,
                a_lp1=a_lp1,
                b_lp1=b_lp1,
                ridge=ridge,
            )
            profile.append(phi_values["phi_signed"])
        results[spec["name"]] = {
            "family": spec["family"],
            "seed": spec["seed"],
            "phi_signed_profile": profile,
            "phi_global": float(np.mean(profile)) if profile else 0.0,
        }
    return results


def run_partition_robustness(args: argparse.Namespace) -> tuple[dict, Path]:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    triplets, tokenizer, cache, projection_matrix, d_proj_info = prepare_experiment(args)
    effective_d_proj = int(d_proj_info["effective_d_proj"])
    projector = create_projector(
        projector_name=args.projector,
        projection_matrix=projection_matrix,
        output_dim=effective_d_proj,
        seed=args.random_seed,
    )
    partition_specs = build_partition_specs(
        d_proj=effective_d_proj,
        base_seed=args.random_seed,
        num_random_partitions=args.num_random_partitions,
    )

    partition_names = [spec["name"] for spec in partition_specs]
    triplet_scalars_by_partition: Dict[str, List[float]] = {name: [] for name in partition_names}
    triplet_results = []

    for triplet in triplets:
        prompt_results = {}
        triplet_partition_scalars: Dict[str, List[float]] = {name: [] for name in partition_names}
        for prompt_key in PROMPT_KEYS:
            prompt_cache = cache[triplet["id"]][prompt_key]
            projected_layers, sample_count = build_projected_layers(
                prompt_cache=prompt_cache,
                tokenizer=tokenizer,
                projector=projector,
                aggregation=args.aggregation,
                window_size=args.window_size,
            )
            profiles_by_partition = compute_partition_profiles(
                projected_layers=projected_layers,
                partition_specs=partition_specs,
                ridge=args.ridge,
            )
            partition_globals = [item["phi_global"] for item in profiles_by_partition.values()]
            profile_vectors = [item["phi_signed_profile"] for item in profiles_by_partition.values()]
            prompt_results[prompt_key] = {
                "sample_count": sample_count,
                "profiles_by_partition": profiles_by_partition,
                "partition_variance_global": float(np.var(partition_globals)),
                "partition_mean_global": float(np.mean(partition_globals)),
                "partition_profile_cosine_mean": mean_pairwise_cosine(profile_vectors),
            }
            for partition_name, partition_result in profiles_by_partition.items():
                triplet_partition_scalars[partition_name].append(partition_result["phi_global"])

        triplet_scalar_summary = {
            partition_name: float(np.mean(values))
            for partition_name, values in triplet_partition_scalars.items()
        }
        for partition_name, scalar in triplet_scalar_summary.items():
            triplet_scalars_by_partition[partition_name].append(scalar)

        triplet_results.append(
            {
                "id": triplet["id"],
                "prompt_results": prompt_results,
                "triplet_phi_global_by_partition": triplet_scalar_summary,
                "triplet_partition_variance_global": float(
                    np.var(list(triplet_scalar_summary.values()))
                ),
            }
        )

    partition_variance_global = float(
        np.mean([item["triplet_partition_variance_global"] for item in triplet_results])
    )
    task_family_variance_global = float(
        np.mean(
            [
                np.var(partition_values)
                for partition_values in triplet_scalars_by_partition.values()
            ]
        )
    )
    robustness_ratio_global = (
        float(partition_variance_global / task_family_variance_global)
        if task_family_variance_global > 0
        else float("inf")
    )

    prompt_partition_variances = []
    prompt_partition_cosines = []
    for triplet_result in triplet_results:
        for prompt_key in PROMPT_KEYS:
            prompt_partition_variances.append(
                triplet_result["prompt_results"][prompt_key]["partition_variance_global"]
            )
            prompt_partition_cosines.append(
                triplet_result["prompt_results"][prompt_key]["partition_profile_cosine_mean"]
            )

    payload = {
        "config": {
            "model_path": args.model_path,
            "triplets_path": args.triplets_path,
            "aggregation": args.aggregation,
            "projector": args.projector,
            "requested_d_proj": args.d_proj,
            "effective_d_proj": effective_d_proj,
            "min_prompt_sample_count": d_proj_info["min_prompt_sample_count"],
            "max_prompt_sample_count": d_proj_info["max_prompt_sample_count"],
            "mean_prompt_sample_count": d_proj_info["mean_prompt_sample_count"],
            "window_size": args.window_size,
            "ridge": args.ridge,
            "random_seed": args.random_seed,
            "num_random_partitions": args.num_random_partitions,
            "device": args.device,
        },
        "summary": {
            "triplet_count": len(triplet_results),
            "partition_count": len(partition_specs),
            "mean_prompt_partition_variance_global": float(np.mean(prompt_partition_variances)),
            "mean_prompt_partition_profile_cosine": float(np.mean(prompt_partition_cosines)),
            "partition_variance_global": partition_variance_global,
            "task_family_variance_global": task_family_variance_global,
            "robustness_ratio_global": robustness_ratio_global,
        },
        "partition_specs": [
            {"name": spec["name"], "family": spec["family"], "seed": spec["seed"]}
            for spec in partition_specs
        ],
        "results": triplet_results,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output_path


def print_summary(payload: dict, output_path: Path) -> None:
    summary = payload["summary"]
    print(f"Partition robustness results saved to: {output_path}")
    print(
        "Summary:"
        f" partitions={summary['partition_count']},"
        f" triplets={summary['triplet_count']},"
        f" requested_d_proj={payload['config']['requested_d_proj']},"
        f" effective_d_proj={payload['config']['effective_d_proj']},"
        f" mean_prompt_partition_variance_global={summary['mean_prompt_partition_variance_global']:.4f},"
        f" mean_prompt_partition_profile_cosine={summary['mean_prompt_partition_profile_cosine']:.4f},"
        f" partition_variance_global={summary['partition_variance_global']:.4f},"
        f" task_family_variance_global={summary['task_family_variance_global']:.4f},"
        f" robustness_ratio_global={summary['robustness_ratio_global']:.4f}"
    )
    for item in payload["results"]:
        print(
            f"  - {item['id']}:"
            f" triplet_partition_variance_global={item['triplet_partition_variance_global']:.4f}"
        )


def main() -> None:
    args = parse_args()
    payload, output_path = run_partition_robustness(args)
    if not args.quiet:
        print_summary(payload, output_path)


if __name__ == "__main__":
    main()
