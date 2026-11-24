"""Create the ten small, executable case notebooks without duplicating the engine."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import nbformat as nbf
    destination = ROOT / "notebooks" / "cases"
    destination.mkdir(parents=True, exist_ok=True)
    for config_path in sorted((ROOT / "configs" / "cases").glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        case_id, title = config["case_id"], config["title"]
        path = destination / f"{case_id}.ipynb"
        if path.exists():
            continue  # An executed notebook is evidence and is never overwritten.
        nb = nbf.v4.new_notebook()
        nb.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                           language_info={"name": "python", "version": "3.12"},
                           ai_buckingham={"case_id": case_id, "execution": "not_yet_executed"})
        nb.cells = [
            nbf.v4.new_markdown_cell(f"# {title}\n\nThis notebook executes the real LangGraph workflow with the supplied case prompt and overrides. All stage cards and final plots are retained below. New execution results are recorded separately from historical evidence.\n\nSet `OPENAI_API_KEY` and `PARATERA_API_KEY` in the kernel environment. See the repository README for endpoint configuration. Credentials never belong in notebook cells."),
            nbf.v4.new_code_cell("from pathlib import Path\nimport sys\n\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs' / 'cases').is_dir() and (p / 'pyproject.toml').is_file())\nsys.path.insert(0, str(ROOT / 'src'))\n\nfrom ai_buckingham import build_agent_graph\nfrom ai_buckingham.cli import load_case, run_analysis\nfrom ai_buckingham.artifacts import new_run_id, read_json\nfrom IPython.display import display, Markdown\nimport pandas as pd"),
            nbf.v4.new_code_cell(f"CASE_ID = {case_id!r}\ncase = load_case(CASE_ID)\nRUN_DIR = ROOT / 'results' / 'cases' / CASE_ID / new_run_id()\ndisplay(Markdown(case['prompt']))\nframe = pd.read_csv(ROOT / case['dataset'])\nprint(f'Original observations: {{len(frame)}}; columns: {{list(frame.columns)}}')\ndisplay(frame.head())\nprint('Declared manual overrides:')\ndisplay(case['manual_hyperparams'])"),
            nbf.v4.new_code_cell("# Build the same nine-node graph used by the CLI and Streamlit.\n# This step does not initialize models or make API calls.\nagent_graph = build_agent_graph()\nprint(agent_graph.get_graph().draw_mermaid())"),
            nbf.v4.new_markdown_cell("## Execute the complete analysis\n\nThe following cell initializes the configured model clients, invokes the graph, records node events and actual optimizer generations, and displays every captured stage card. It can take substantial time for the full numerical settings. Changing a scientific setting creates a new run; it does not overwrite this result."),
            nbf.v4.new_code_cell("state = run_analysis(CASE_ID, output_dir=RUN_DIR, agent_graph=agent_graph)\nmanifest = read_json(RUN_DIR / 'manifest.json')\nprint('Workflow status:', state.get('workflow_status'))\nprint('Evidence directory:', RUN_DIR.relative_to(ROOT).as_posix())\nassert manifest['status'] == 'completed', 'Inspect recorded diagnostics: this run did not complete successfully.'"),
            nbf.v4.new_code_cell("# Numeric exports are the common source for reports and the web viewer.\nfor table in sorted((RUN_DIR / 'tables').glob('*.csv')):\n    display(Markdown(f'**{table.name}**'))\n    display(pd.read_csv(table).head(30))\nprint('Final figures:', [p.name for p in sorted((RUN_DIR / 'figures').glob('*'))])\nprint('Actual optimizer trace:', (RUN_DIR / 'evidence' / 'optimization_trace.jsonl').relative_to(ROOT).as_posix())"),
            nbf.v4.new_markdown_cell("## Interpretation and provenance\n\nThe complete state, node events, retrieval evidence, resolved configuration, environment versions and source hashes are stored beside the standalone HTML report. Conditional collapse-validation scores assess the selected representation; they do not establish out-of-sample performance of the complete discovery process. Optimizer objective curves may include penalties and must not automatically be relabeled as mutual information."),
        ]
        nbf.write(nb, path)
        print(path.relative_to(ROOT).as_posix())
    quickstart = ROOT / "notebooks" / "quickstart.ipynb"
    if not quickstart.exists():
        nb = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}})
        nb.cells = [
            nbf.v4.new_markdown_cell("# AI Buckingham: offline evidence quickstart\n\nRun all cells without API credentials. This notebook reads included datasets, numerical summaries and one recorded result. It does not initialize models or repeat discovery. For a new complete execution, use a notebook in `cases/` after configuring credentials."),
            nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nimport json\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs' / 'cases').is_dir() and (p / 'pyproject.toml').is_file())\nsys.path.insert(0, str(ROOT / 'src'))\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nfrom ai_buckingham.artifacts import find_runs, load_run"),
            nbf.v4.new_code_cell("catalog = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT / 'configs' / 'cases').glob('*.json'))]\ndisplay(pd.DataFrame([{'case_id':c['case_id'], 'title':c['title'], 'rows':len(pd.read_csv(ROOT / c['dataset']))} for c in catalog]))"),
            nbf.v4.new_code_cell("summary = pd.read_csv(ROOT / 'results' / 'summary.csv')\ndisplay(summary)\nprint('Conditional held-out R² evaluates a selected representation, not the complete discovery process.')"),
            nbf.v4.new_code_cell("completed = [r for r in find_runs(ROOT, 'mhd') if r['status'] == 'completed']\nrun = load_run(completed[-1])\nprint('Run:', run['manifest']['run_id'])\nprint('Source:', run['manifest']['interpretation'])\nfor local in run['state']['workflow_result']['local_workflows']:\n    display(Markdown('**Selected relation:** ' + '; '.join(local['final_solution']['input_formulas'])))\nfor figure in run['figures']:\n    if figure.endswith('.png'):\n        display(Image(filename=figure))"),
            nbf.v4.new_markdown_cell("Open the [hosted Streamlit evidence explorer](https://ai-buckingham.streamlit.app/) to browse retrieval scores, candidate partitions, numerical diagnostics, actual CMA-ES generations, comparisons and full result cards. For a local session, open `streamlit_app.py` with `streamlit run streamlit_app.py`. The ten case notebooks contain the complete graph executions."),
        ]
        nbf.write(nb, quickstart)
        print(quickstart.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
