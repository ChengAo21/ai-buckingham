# Mutual-information landscapes

The original unexecuted notebook retains numerical definitions and settings for four examples in Main Fig. 4. The CLI uses the central `data/datasets/` inputs with public case IDs and leaves the notebook source unchanged.

From the repository root, after installing `environments/mi-landscapes.txt`:

```bash
python reproduction/mi_landscapes/reproduce.py
```

This starts fresh CMA-ES searches with seed 42 and a 101 x 101 objective grid. `--case` accepts `rayleigh_benard`, `pressure_gradient`, `laser_keyhole`, or `blasius`. Outputs go to `generated/optimize/<case_id>/` within this package unless `--output-dir` is provided.

Each run exports the numeric grid, optimization history, basis-selection table, coefficients, input hashes, environment metadata and rendered landscape. Search trajectories and slice selection can vary with optimizer and numerical-library behavior. Fresh runs are not represented as byte-identical historical figure trajectories.

For a reduced execution check:

```bash
python reproduction/mi_landscapes/reproduce.py --case blasius --maxiter 2 --popsize 8 --grid-size 15
```

The reduced settings are recorded in run metadata and must not be reported as the full manuscript optimization.
