"""Regenerate four case-study CSVs with the archived sampling parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import root


def _blasius_rhs(state: np.ndarray, _: float) -> np.ndarray:
    f, first, second = state
    return np.array([first, second, -f * second / 2])


def generate_blasius() -> pd.DataFrame:
    eta_grid = np.arange(0, 10, 0.01)

    def boundary_residual(initial: np.ndarray) -> list[float]:
        solution = odeint(_blasius_rhs, initial, eta_grid)
        return [initial[0], initial[1], solution[-1, 1] - 1]

    match = root(boundary_residual, [0, 0, 0], tol=1e-4)
    initial = [0, 0, match.x[2]]
    free_stream_velocity = 0.01
    viscosity = 1e-6
    x = np.linspace(1e-3, 1e-1, 100)
    delta = 1.72 * np.sqrt(x * viscosity / free_stream_velocity)
    y = np.linspace(1e-4, 2 * max(delta), 100)
    yy, xx = np.meshgrid(y, x)
    eta = yy * np.sqrt(free_stream_velocity / (xx * viscosity))

    # Integrate on each sampled profile's grid, as in the archived calculation.
    velocity = np.zeros(eta.shape)
    for i, profile in enumerate(eta):
        solution = odeint(_blasius_rhs, initial, profile)
        velocity[i, :] = free_stream_velocity * solution[:, 1]

    frame = pd.DataFrame(
        {
            "U": free_stream_velocity,
            "nu": viscosity,
            "x": xx.flatten(),
            "y": yy.flatten(),
            "u/U": velocity.flatten() / free_stream_velocity,
        }
    )
    return frame.sample(frac=0.02, random_state=5)


def generate_goldenfeld() -> pd.DataFrame:
    observations = pd.read_csv(Path(__file__).with_name("nikuradse_data.csv"))
    # RandomState retains the draw sequence from the original notebook.
    random = np.random.RandomState(42)
    count = len(observations)
    density = random.uniform(1.0, 1000.0, count)
    viscosity = random.uniform(1e-5, 1e-2, count)
    diameter = random.uniform(0.1, 5.0, count)
    return pd.DataFrame(
        {
            "rho": density,
            "mu": viscosity,
            "D": diameter,
            "U": observations["Re"].to_numpy() * viscosity / (density * diameter),
            "r": observations["r/D"].to_numpy() * diameter,
            "case": (1 / observations["r/D"]).round(2),
            "cf": observations["f"].to_numpy(),
        }
    )


def _log_uniform(
    random: np.random.Generator, low: float, high: float, count: int
) -> np.ndarray:
    return np.exp(random.uniform(np.log(low), np.log(high), size=count))


def _colebrook_friction_factor(
    reynolds: np.ndarray, relative_roughness: np.ndarray
) -> np.ndarray:
    # Newton iteration in inverse square-root friction factor, initialized with Haaland.
    x = -1.8 * np.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds)
    x = np.maximum(x, 1e-8)
    for _ in range(80):
        argument = relative_roughness / 3.7 + 2.51 * x / reynolds
        residual = x + 2.0 * np.log10(argument)
        derivative = 1.0 + (2.0 / np.log(10.0)) * (2.51 / reynolds) / argument
        updated = np.maximum(x - residual / derivative, 1e-8)
        max_step = np.max(np.abs(updated - x))
        x = updated
        argument = relative_roughness / 3.7 + 2.51 * x / reynolds
        max_residual = np.max(np.abs(x + 2.0 * np.log10(argument)))
        if max_step < 1e-12 and max_residual < 1e-11:
            return 1.0 / x**2
    raise RuntimeError(f"Colebrook iteration did not converge (residual {max_residual:.3e}).")


def generate_pipe() -> pd.DataFrame:
    random = np.random.default_rng(42)
    count = 1200
    accepted = []
    accepted_count = 0
    while accepted_count < count:
        batch = max(256, 2 * (count - accepted_count))
        density = random.uniform(0.1, 0.18, batch)
        velocity = _log_uniform(random, 0.025, 800.0, batch)
        viscosity = random.uniform(1.0e-6, 1.0e-5, batch)
        diameter = random.uniform(0.3, 1.0, batch)
        roughness = random.uniform(5.0e-3, 1.0e-1, batch)
        reynolds = density * velocity * diameter / viscosity
        relative_roughness = roughness / diameter
        keep = (
            (208.0 <= reynolds)
            & (reynolds <= 4e7)
            & (0.005 <= relative_roughness)
            & (relative_roughness <= 0.11)
        )
        accepted.append(
            np.column_stack((density, viscosity, diameter, velocity, roughness))[keep]
        )
        accepted_count += int(np.sum(keep))

    frame = pd.DataFrame(
        np.concatenate(accepted)[:count], columns=["rho", "mu", "D", "U", "epsilon"]
    )
    density, viscosity, diameter, velocity, roughness = frame.to_numpy().T
    reynolds = density * velocity * diameter / viscosity
    relative_roughness = roughness / diameter
    laminar = reynolds <= 3000.0
    friction = np.empty(count, dtype=float)
    friction[laminar] = 64.0 / reynolds[laminar]
    friction[~laminar] = _colebrook_friction_factor(
        reynolds[~laminar], relative_roughness[~laminar]
    )
    frame["f_D"] = friction
    return frame


def generate_cess() -> pd.DataFrame:
    random = np.random.default_rng(30)
    count = 1000
    friction_velocity = _log_uniform(random, 1.0e-2, 1.0e3, count)
    half_height = random.uniform(0.5, 1.0, count)
    viscosity = random.uniform(1.0e-6, 1.0e-5, count)

    # Match the archived sampling proportions across three wall-distance bands.
    relative_y = np.concatenate(
        [
            _log_uniform(random, 1.0e-6, 0.148, 789),
            _log_uniform(random, 0.148, 0.368, 105),
            _log_uniform(random, 0.368, 1.0, 106),
        ]
    )
    random.shuffle(relative_y)
    y = half_height * relative_y

    eta = y / half_height
    friction_reynolds = friction_velocity * half_height / viscosity
    y_plus = friction_reynolds * eta
    one_minus_eta_squared = (1.0 - eta) ** 2
    shape = (
        (1.0 - one_minus_eta_squared) ** 2
        * (1.0 + 2.0 * one_minus_eta_squared) ** 2
        * (1.0 - np.exp(-y_plus / 25.4)) ** 2
    )
    inside = 1.0 + (0.426**2 * friction_reynolds**2 / 9.0) * shape
    ratio = 0.5 * (np.sqrt(inside) - 1.0)
    return pd.DataFrame(
        {"u_tau": friction_velocity, "h": half_height, "y": y, "nu": viscosity, "nu_t/nu": ratio}
    )


def main() -> None:
    generators = {
        "blasius": generate_blasius,
        "goldenfeld": generate_goldenfeld,
        "rough_pipe": generate_pipe,
        "cess_channel": generate_cess,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("case", nargs="*", choices=tuple(generators))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    files = {key: output_dir / f"{key}.csv" for key in args.case or generators}
    for path in files.values():
        if path.exists():
            parser.error(f"Output already exists: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for key, path in files.items():
        frame = generators[key]()
        frame.to_csv(path, index=False, mode="x", lineterminator="\r\n")
        print(f"{path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
