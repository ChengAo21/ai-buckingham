# MI optimization landscapes

This package generates MI optimization landscapes for the MHD power generator and the four Type I cases in Figure 4. Each case uses the notebook's CMA-ES search and MI estimator with the corresponding inputs in `data/datasets/`.

From the repository root, after installing `environments/mi-landscapes.txt`:

```bash
python reproduction/mi_landscapes/reproduce.py
```

The command runs the four Figure 4 cases with seed 42 and evaluates a 101 × 101 grid for each. To run the MHD case with the same notebook, use the following command.

```bash
python reproduction/mi_landscapes/reproduce.py --case mhd
```

To run a single Figure 4 case, pass `--case pressure_gradient`, `--case rayleigh_benard`, `--case laser_keyhole` or `--case blasius`. Outputs go to `generated/optimize/<case_id>/` within this package unless `--output-dir` is provided.

Each run exports the grid, optimization history, basis-selection table, coefficients and figures.
