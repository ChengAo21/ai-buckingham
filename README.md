<div align="center">

# AI Buckingham: Agentic Discovery of Dimensionless Scaling Laws Across Physical Regimes

[![Results explorer](https://img.shields.io/badge/Results-Open_explorer-087f8c?style=flat-square)](https://chengao21.github.io/ai-buckingham/)
[![LangGraph notebooks](https://img.shields.io/badge/LangGraph-10_notebooks-245b78?style=flat-square)](notebooks/cases/)
[![Streamlit app](https://img.shields.io/badge/Streamlit-Open_app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://ai-buckingham.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-536e70?style=flat-square)](pyproject.toml)
[![MIT License](https://img.shields.io/badge/License-MIT-d38b50?style=flat-square)](LICENSE)

[Overview](#research-overview) · [Workflow](#workflow) · [Ten physical systems](#ten-physical-systems) · [Run locally](#run-locally) · [Reproduce](#reproduce-and-verify) · [Citation](#citation-and-terms)

</div>

## Research overview

AI Buckingham is an agentic framework for autonomous dimensionless discovery from natural-language descriptions of physical problems and observational data. It couples LLM-based reasoning and planning with symbolic and numerical tools for dimensional checking, physical-regime identification, dimensionless learning, simplification and validation.

**Ao Cheng, Lei Zhang and Guowei He**

The accompanying paper studies ten physical systems: six cases of **Type I similarity: complete similarity**, two cases of **Type II similarity: incomplete similarity**, and two **Multi-regime physical systems**.

## Workflow

![Schematic of the AI Buckingham workflow](docs/assets/workflow.png)

Figure 2 of the paper organizes the workflow into three layers:

- **Semantic-planning layer:** parse the problem specification, check variable dimensions, construct the Buckingham–Π basis and select the analysis route.
- **Numerical-inference layer:** perform physical-regime and similarity-type screening when required, MI-based dimensionless learning, adaptive group-count selection, sparsification, rationalization and held-out validation.
- **Synthesis-and-reporting layer:** summarize the discovered scaling relations and regime-wise data collapse.

The two literature-based knowledge bases contain 247 physical-quantity entries and 251 dimensionless-relation entries for dimensional checking and physical interpretation.

## Ten physical systems

| Similarity setting | Case | Observations | Notebook | Results | Figure |
|---|---|---:|---|---|---|
| Type I | Magnetohydrodynamic power generator | 483 | [Notebook](notebooks/cases/mhd.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/mhd.html) | [Figure](docs/assets/cases/mhd.png) |
| Type I | Skin friction under pressure gradients | 144 | [Notebook](notebooks/cases/pressure_gradient.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/pressure_gradient.html) | [Figure](docs/assets/cases/pressure_gradient.png) |
| Type I | Rayleigh–Bénard convection | 182 | [Notebook](notebooks/cases/rayleigh_benard.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/rayleigh_benard.html) | [Figure](docs/assets/cases/rayleigh_benard.png) |
| Type I | Laser-metal interaction in additive manufacturing | 90 | [Notebook](notebooks/cases/laser_keyhole.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/laser_keyhole.html) | [Figure](docs/assets/cases/laser_keyhole.png) |
| Type I | Blasius laminar boundary layer | 200 | [Notebook](notebooks/cases/blasius.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/blasius.html) | [Figure](docs/assets/cases/blasius.png) |
| Type I | Sand-grain roughness parameterization | 104 | [Notebook](notebooks/cases/sandgrain.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/sandgrain.html) | [Figure](docs/assets/cases/sandgrain.png) |
| Type II | Velocity scaling in compressible wall-bounded turbulence | 1,109 | [Notebook](notebooks/cases/velocity_scaling.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/velocity_scaling.html) | [Figure](docs/assets/cases/velocity_scaling.png) |
| Type II | Goldenfeld’s roughness-induced criticality in pipe flow | 414 | [Notebook](notebooks/cases/goldenfeld.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/goldenfeld.html) | [Figure](docs/assets/cases/goldenfeld.png) |
| Multi-regime | Cess eddy-viscosity model in turbulent channel flows | 1,000 | [Notebook](notebooks/cases/cess_channel.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/cess_channel.html) | [Figure](docs/assets/cases/cess_channel.png) |
| Multi-regime | Viscous flow in a rough circular pipe | 1,200 | [Notebook](notebooks/cases/rough_pipe.ipynb) | [Results](https://chengao21.github.io/ai-buckingham/cards/rough_pipe.html) | [Figure](docs/assets/cases/rough_pipe.png) |

Dataset sources, preparation and analysis-eligible sample counts are documented in [data/README.md](data/README.md).

## Interactive explorer

[Open the results explorer](https://chengao21.github.io/ai-buckingham/) to browse all ten cases, or [open the Streamlit application](https://ai-buckingham.streamlit.app/) to inspect the workflow and run an analysis. Browsing saved results requires no API credentials. New analyses require configured LLM and embedding services.

See the [application guide](docs/app.md) for available views and model configuration.

## Run locally

Use **Python 3.12**, then run from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e ".[app,notebooks,dev]"
python -m streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), or use the [offline quickstart notebook](notebooks/quickstart.ipynb).

<details>
<summary><strong>Run an analysis and configure model access</strong></summary>

Set credentials in the process environment using the names in [`.env.example`](.env.example). The application does not automatically load credential files. The default model is GPT-5 mini; embeddings use GLM-Embedding-3. Configure compatible endpoints through `OPENAI_BASE_URL` and `PARATERA_BASE_URL`.

```bash
ai-buckingham --list
ai-buckingham mhd
```

```python
from ai_buckingham import run_case

state = run_case("mhd")
```

By default, each run saves its configuration, results, figures and execution trace under `generated/cases/<case_id>/<run_id>/`. Custom configurations can be passed to `ai_buckingham.cli.run_analysis`. Case settings and numerical thread profiles are described in the [configuration guide](docs/configuration.md).

Full-budget analyses can take substantial time.

</details>

## Reproduce and verify

The repository includes executed notebooks and saved results for all ten cases. From the repository root, run:

```bash
python -m pytest -q
python scripts/check_publication.py
python scripts/validate_results.py
python scripts/check_app.py
python scripts/check_web.py
python reproduction/verify_integrity.py
```

| Reproduction package | Paper content |
|---|---|
| [Dimensional checking](reproduction/rag_ablation/) | LLM ablation and API usage in dimensional checking (SI §5.2) |
| [Velocity transformations](reproduction/velocity_comparison/) | Relation to the Trettel–Larsson transformation (SI §5.3) |
| [Physical-regime comparison](reproduction/regime_comparison/) | Physical-regime detection in IT–π and AI Buckingham (SI §5.4) |
| [MI optimization landscapes](reproduction/mi_landscapes/) | CMA-ES trajectories and MI grids for MHD and the four Type I cases in Figure 4 |

See the [reproduction guide](reproduction/README.md) for commands and dependencies, [result files](docs/evidence.md) for file contents, and [repository layout](docs/repository-layout.md) for directory navigation.

## Citation and terms

If you use AI Buckingham in your research, please cite the following paper:

Ao Cheng, Lei Zhang and Guowei He. *AI Buckingham: Agentic Discovery of Dimensionless Scaling Laws Across Physical Regimes*. arXiv preprint arXiv:2611.33961 (2026). [doi:10.48550/arXiv.2611.33961](https://doi.org/10.48550/arXiv.2611.33961).

```bibtex
@article{cheng2026aibuckingham,
  title   = {{AI Buckingham}: Agentic Discovery of Dimensionless Scaling Laws Across Physical Regimes},
  author  = {Ao Cheng and Lei Zhang and Guowei He},
  journal = {arXiv preprint arXiv:2611.33961},
  year    = {2026},
  doi     = {10.48550/arXiv.2611.33961}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff). Please acknowledge the original dataset sources you use. Original project code is released under the [MIT License](LICENSE); third-party terms are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Acknowledgements

We thank GPT-6.2 (OpenAI), via Codex, for assistance with reproducibility checks, documentation editing and refinements to the results explorer interfaces. The authors reviewed the assisted work and retain full responsibility. We also thank the developers of the open-source tools and the contributors of the datasets acknowledged in this repository.
