import argparse
import json
import os
from pathlib import Path

MPL_CONFIG_DIR = Path("./outputs/.mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT_PATH = "./outputs/phi_proxy_results.json"
DEFAULT_OUTPUT_DIR = "./outputs/figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phi profiles and stability curves from Phi-proxy results."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_triplet(result: dict, output_dir: Path) -> Path:
    triplet_id = result["id"]
    prompt_results = result["prompt_results"]

    phi_original = np.asarray(prompt_results["original"]["phi_signed_profile"], dtype=float)
    phi_surface = np.asarray(prompt_results["surface"]["phi_signed_profile"], dtype=float)
    phi_inverse = np.asarray(prompt_results["inverse"]["phi_signed_profile"], dtype=float)
    layers = np.arange(len(phi_original))

    phi_stability = np.asarray(result["phi_stability"]["stability_by_layer"], dtype=float)
    surface_stability = np.asarray(
        result["surface_stability"]["stability_by_layer"], dtype=float
    )
    inverse_stability = np.asarray(
        result["inverse_stability"]["stability_by_layer"], dtype=float
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].plot(layers, phi_original, marker="o", linewidth=2, label="original")
    axes[0].plot(layers, phi_surface, marker="s", linewidth=2, label="surface")
    axes[0].plot(layers, phi_inverse, marker="^", linewidth=2, label="inverse")
    axes[0].set_title("Phi by Layer")
    axes[0].set_xlabel("Layer Transition")
    axes[0].set_ylabel("Phi-proxy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(layers, phi_stability, marker="o", linewidth=2, label="global")
    axes[1].plot(layers, surface_stability, marker="s", linewidth=2, label="surface")
    axes[1].plot(layers, inverse_stability, marker="^", linewidth=2, label="inverse")
    axes[1].set_title("Stability by Layer")
    axes[1].set_xlabel("Layer Transition")
    axes[1].set_ylabel("Stability")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        f"{triplet_id} | "
        f"phi={result['phi_stability']['stability_global']:.3f} | "
        f"surface={result['surface_stability']['stability_global']:.3f} | "
        f"inverse={result['inverse_stability']['stability_global']:.3f}",
        fontsize=12,
    )

    output_path = output_dir / f"{triplet_id}.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_overview(payload: dict, output_dir: Path) -> Path:
    results = payload["results"]
    rows = len(results)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), constrained_layout=True)

    if rows == 1:
        axes = np.asarray([axes])

    for row, result in enumerate(results):
        prompt_results = result["prompt_results"]
        phi_original = np.asarray(prompt_results["original"]["phi_signed_profile"], dtype=float)
        phi_surface = np.asarray(prompt_results["surface"]["phi_signed_profile"], dtype=float)
        phi_inverse = np.asarray(prompt_results["inverse"]["phi_signed_profile"], dtype=float)
        layers = np.arange(len(phi_original))

        phi_stability = np.asarray(result["phi_stability"]["stability_by_layer"], dtype=float)
        surface_stability = np.asarray(
            result["surface_stability"]["stability_by_layer"], dtype=float
        )
        inverse_stability = np.asarray(
            result["inverse_stability"]["stability_by_layer"], dtype=float
        )

        left = axes[row, 0]
        right = axes[row, 1]

        left.plot(layers, phi_original, marker="o", linewidth=2, label="original")
        left.plot(layers, phi_surface, marker="s", linewidth=2, label="surface")
        left.plot(layers, phi_inverse, marker="^", linewidth=2, label="inverse")
        left.set_title(f"{result['id']} | Phi")
        left.set_xlabel("Layer Transition")
        left.set_ylabel("Phi-proxy")
        left.grid(alpha=0.3)
        left.legend()

        right.plot(layers, phi_stability, marker="o", linewidth=2, label="global")
        right.plot(layers, surface_stability, marker="s", linewidth=2, label="surface")
        right.plot(layers, inverse_stability, marker="^", linewidth=2, label="inverse")
        right.set_title(f"{result['id']} | Stability")
        right.set_xlabel("Layer Transition")
        right.set_ylabel("Stability")
        right.set_ylim(0.0, 1.05)
        right.grid(alpha=0.3)
        right.legend()

    output_path = output_dir / "phi_profiles_overview.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(input_path)
    generated_paths = [plot_triplet(result, output_dir) for result in payload["results"]]
    overview_path = plot_overview(payload, output_dir)

    print(f"Plots saved to: {output_dir}")
    print(f"Overview: {overview_path}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
