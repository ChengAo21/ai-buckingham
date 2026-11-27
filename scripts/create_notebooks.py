"""Create notebooks that render saved results and offer an explicit new analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = """from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'configs/cases').is_dir() and (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT / 'src'))

from IPython.display import display, HTML, Image
from ai_buckingham import build_agent_graph
from ai_buckingham.artifacts import new_run_id
from ai_buckingham.cli import run_analysis
from ai_buckingham.notebook import published_run
from ai_buckingham.reports import render_report, export_report"""


def case_notebook(case: dict):
    import nbformat as nbf

    case_id = case["id"]
    nb = nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "ai_buckingham": {"case_id": case_id, "execution": "not_yet_executed", "execution_mode": "saved_results"},
    })
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# {case['title']}\n\n"
            "Run the cells below to generate the case report from the paper's saved results. "
            "The report shows the system's completed analysis cards and result figures. "
            "Report generation uses local files.\n\n"
            "The final section runs a new analysis from the case input data and configuration "
            "when `RUN_NEW_ANALYSIS` is enabled."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(f"CASE_ID = {case_id!r}\nRUN_DIR = published_run(CASE_ID, ROOT)"),
        nbf.v4.new_markdown_cell("## Executable workflow\n\nLangGraph draws this diagram using its native Mermaid renderer. Rerunning this cell requires Internet access to the rendering service."),
        nbf.v4.new_code_cell("agent_graph = build_agent_graph()\ndisplay(Image(agent_graph.get_graph().draw_mermaid_png()))"),
        nbf.v4.new_markdown_cell("## Case results"),
        nbf.v4.new_code_cell("RESULT_HTML = render_report(RUN_DIR)\ndisplay(HTML(RESULT_HTML))"),
        nbf.v4.new_markdown_cell(
            "## Save the HTML report\n\nSet `SAVE_HTML = True` to write a standalone copy under `generated/reports/`. "
            "Open the resulting `result.html` in a browser."),
        nbf.v4.new_code_cell(
            "SAVE_HTML = False\n"
            "if SAVE_HTML:\n"
            "    report_path = export_report(RUN_DIR, ROOT / 'generated' / 'reports' / CASE_ID / 'result.html')"),
        nbf.v4.new_markdown_cell(
            "## Run a new analysis\n\n"
            f"The input data and analysis settings are in [the {case_id} configuration](../../configs/cases/{case_id}.json). "
            "Configure the model and embedding credentials described in [the README](../../README.md#run-locally), "
            "then set `RUN_NEW_ANALYSIS = True`. This runs the full workflow and can take substantial time. "
            "The new cards and files are saved in a separate `generated/cases/` run directory.\n\n"
            f"The command-line equivalent from the repository root is `ai-buckingham {case_id}`."),
        nbf.v4.new_code_cell(
            "RUN_NEW_ANALYSIS = False\n"
            "if RUN_NEW_ANALYSIS:\n"
            "    NEW_RUN_DIR = ROOT / 'generated' / 'cases' / CASE_ID / new_run_id()\n"
            "    state = run_analysis(CASE_ID, output_dir=NEW_RUN_DIR, agent_graph=agent_graph, display_cards=False)\n"
            "    display(HTML(render_report(NEW_RUN_DIR)))"),
    ]
    for index, cell in enumerate(nb.cells):
        cell.id = f"{case_id}-{index:02d}"
    return nb


def quickstart_notebook():
    import nbformat as nbf

    nb = nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# AI Buckingham\n\nGenerate the MHD HTML report from the saved paper results. "
            "These cells run offline. Choose another case by changing `CASE_ID`; the ten case notebooks "
            "also include instructions for running a new analysis."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell("CASE_ID = 'mhd'\nRUN_DIR = published_run(CASE_ID, ROOT)\ndisplay(HTML(render_report(RUN_DIR)))"),
        nbf.v4.new_markdown_cell(
            "[Open the MHD reproduction notebook](cases/mhd.ipynb) or "
            "[browse all ten cases in Streamlit](https://ai-buckingham.streamlit.app/)."),
    ]
    for index, cell in enumerate(nb.cells):
        cell.id = f"quickstart-{index:02d}"
    return nb


def render_saved_outputs(nb):
    """Execute report and native graph cells with the IPython display system."""
    import nbformat as nbf
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    try:
        for index, cell in enumerate((c for c in nb.cells if c.cell_type == "code"), 1):
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise result.error_before_exec or result.error_in_exec
            cell.execution_count = index
            cell.outputs = [nbf.v4.new_output("display_data", data=output.data, metadata=output.metadata)
                            for output in captured.outputs]
            if captured.stderr.strip():
                cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=captured.stderr))
        if "ai_buckingham" in nb.metadata:
            case_id = nb.metadata.ai_buckingham.case_id
            index = json.loads((ROOT / "results/index.json").read_text(encoding="utf-8"))
            case = next(item for item in index["cases"] if item["case_id"] == case_id)
            run = next(item for item in case["runs"] if item["run_id"] == case["selected_run"])
            nb.metadata.ai_buckingham.update(execution="completed", run_directory=run["directory"])
    finally:
        shell.clear_instance()
    return nb


def main():
    import nbformat as nbf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace existing notebook views using the current template")
    parser.add_argument("--render", action="store_true", help="Execute the report and native graph cells and save their outputs")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "docs/paper_cases.json").read_text(encoding="utf-8"))
    destination = ROOT / "notebooks/cases"
    destination.mkdir(parents=True, exist_ok=True)
    notebooks = [(destination / f"{case['id']}.ipynb", case_notebook(case)) for case in metadata["cases"]]
    notebooks.append((ROOT / "notebooks/quickstart.ipynb", quickstart_notebook()))
    for path, nb in notebooks:
        if path.exists() and not args.refresh:
            continue
        if args.render:
            render_saved_outputs(nb)
        nbf.write(nb, path)
        print(path.relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
