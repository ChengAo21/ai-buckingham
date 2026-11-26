"""Presentation extracted from the author implementation."""
from __future__ import annotations

import os
import json
import numpy as np
from IPython.display import display, HTML, Image
from datetime import datetime
from typing import List, Any, Optional
import html
from . import config as _config
from . import optimization as _optimization
from . import plotting as _plotting
from . import provenance as _provenance
from . import workflow as _workflow


_AGENT_EDITORIAL_STYLE_LOADED = False


_AGENT_EDITORIAL_CSS = r"""
<style>
:root {
  --daa-ink: #17202a;
  --daa-muted: #667085;
  --daa-line: #d9dee5;
  --daa-soft: #f6f8fa;
  --daa-paper: #ffffff;
  --daa-blue: #245b78;
  --daa-blue-soft: #edf4f7;
  --daa-green: #3f7652;
  --daa-green-soft: #eef6f0;
  --daa-amber: #9a641f;
  --daa-amber-soft: #fbf4e8;
  --daa-red: #9a3d3d;
  --daa-red-soft: #faeeee;
}
.daa-shell {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  color: var(--daa-ink);
  max-width: 1180px;
  margin: 10px 0 18px 0;
}
.daa-masthead {
  border-top: 4px solid var(--daa-ink);
  border-bottom: 1px solid var(--daa-line);
  padding: 18px 2px 15px 2px;
  margin: 4px 0 15px 0;
}
.daa-kicker {
  color: var(--daa-blue);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: .13em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.daa-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 30px;
  line-height: 1.15;
  font-weight: 600;
  letter-spacing: -.02em;
  margin: 0;
  color: #111820;
}
.daa-deck {
  color: #4b5563;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 15px;
  line-height: 1.5;
  max-width: 980px;
  margin-top: 8px;
}
.daa-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-top: 13px;
  color: var(--daa-muted);
  font-size: 12px;
}
.daa-meta strong { color: var(--daa-ink); font-weight: 650; }
.daa-route {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  border: 1px solid var(--daa-line);
  background: var(--daa-paper);
  margin: 10px 0 18px 0;
}
.daa-route-step {
  min-height: 52px;
  padding: 9px 11px;
  border-right: 1px solid var(--daa-line);
  position: relative;
}
.daa-route-step:last-child { border-right: none; }
.daa-route-step.complete { background: var(--daa-green-soft); }
.daa-route-step.current { background: var(--daa-blue-soft); }
.daa-route-index {
  color: var(--daa-muted);
  font-size: 10px;
  font-weight: 750;
  letter-spacing: .08em;
}
.daa-route-label {
  color: var(--daa-ink);
  font-size: 12px;
  font-weight: 650;
  line-height: 1.25;
  margin-top: 3px;
}
.daa-card {
  border: 1px solid var(--daa-line);
  border-left: 4px solid var(--daa-blue);
  background: var(--daa-paper);
  padding: 14px 16px 13px 16px;
  margin: 10px 0 12px 0;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .035);
}
.daa-card[data-tone="success"] { border-left-color: var(--daa-green); }
.daa-card[data-tone="warning"] { border-left-color: var(--daa-amber); }
.daa-card[data-tone="danger"] { border-left-color: var(--daa-red); }
.daa-card[data-tone="muted"] { border-left-color: #98a2b3; }
.daa-card-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 8px;
}
.daa-card-label {
  color: var(--daa-blue);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.daa-card-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 20px;
  font-weight: 600;
  line-height: 1.22;
  margin: 0;
  color: #151b23;
}
.daa-status {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 4px 7px;
  border: 1px solid var(--daa-line);
  color: var(--daa-muted);
  white-space: nowrap;
}
.daa-status.success { color: var(--daa-green); background: var(--daa-green-soft); border-color: #c9ddcf; }
.daa-status.warning { color: var(--daa-amber); background: var(--daa-amber-soft); border-color: #ead5b5; }
.daa-status.danger { color: var(--daa-red); background: var(--daa-red-soft); border-color: #e8c7c7; }
.daa-status.info { color: var(--daa-blue); background: var(--daa-blue-soft); border-color: #cbdde5; }
.daa-summary {
  font-size: 13px;
  line-height: 1.48;
  color: #4b5563;
  max-width: 1020px;
  margin: 2px 0 10px 0;
}
.daa-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  border-top: 1px solid var(--daa-line);
  border-left: 1px solid var(--daa-line);
  margin-top: 10px;
}
.daa-metric {
  min-height: 51px;
  padding: 8px 10px;
  border-right: 1px solid var(--daa-line);
  border-bottom: 1px solid var(--daa-line);
  background: #fff;
}
.daa-metric-key {
  font-size: 9px;
  line-height: 1.2;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--daa-muted);
}
.daa-metric-value {
  font-size: 13px;
  line-height: 1.3;
  color: var(--daa-ink);
  margin-top: 4px;
  overflow-wrap: anywhere;
}
.daa-formula {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.45;
  color: #243746;
  background: var(--daa-soft);
  border: 1px solid var(--daa-line);
  padding: 8px 10px;
  margin-top: 9px;
  overflow-wrap: anywhere;
}
.daa-note {
  font-family: Georgia, "Times New Roman", serif;
  color: #404854;
  font-size: 13px;
  line-height: 1.48;
  border-left: 2px solid #9fb6c3;
  padding-left: 10px;
  margin-top: 9px;
}
.daa-details { margin-top: 9px; color: #475467; font-size: 12px; }
.daa-details summary { cursor: pointer; font-weight: 700; color: var(--daa-blue); }
.daa-details table { width: 100%; border-collapse: collapse; margin-top: 7px; }
.daa-details td { padding: 5px 7px; border-bottom: 1px solid #e6e9ed; vertical-align: top; }
.daa-details td:first-child { width: 25%; font-weight: 650; color: #475467; }
.daa-result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.daa-result {
  border: 1px solid var(--daa-line);
  padding: 11px 12px;
  background: #fff;
}
.daa-result h4 {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 15px;
  margin: 0 0 7px 0;
}
.daa-result p { margin: 4px 0; font-size: 12px; color: #4b5563; }

.daa-section-title {
  margin: 14px 0 6px 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 14px;
  font-weight: 600;
  color: #1f2933;
}
.daa-diagnostic-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 7px;
  font-size: 11.5px;
}
.daa-diagnostic-table th,
.daa-diagnostic-table td {
  padding: 6px 7px;
  border-bottom: 1px solid #e6e9ed;
  text-align: left;
  vertical-align: top;
}
.daa-diagnostic-table th {
  background: var(--daa-soft);
  color: #475467;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.daa-diagnostic-table td.num,
.daa-diagnostic-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
.daa-mini-badge {
  display: inline-block;
  padding: 2px 6px;
  border: 1px solid var(--daa-line);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .05em;
  text-transform: uppercase;
  white-space: nowrap;
}
.daa-mini-badge.pass { color: var(--daa-green); background: var(--daa-green-soft); border-color: #c9ddcf; }
.daa-mini-badge.fail { color: var(--daa-red); background: var(--daa-red-soft); border-color: #e8c7c7; }
.daa-mini-badge.neutral { color: var(--daa-blue); background: var(--daa-blue-soft); border-color: #cbdde5; }
.daa-diagnostic-note {
  margin-top: 7px;
  color: var(--daa-muted);
  font-size: 10.5px;
  line-height: 1.42;
}
.daa-regime-details {
  margin-top: 7px;
  border-top: 1px solid var(--daa-line);
  padding-top: 6px;
}
.daa-regime-details summary {
  cursor: pointer;
  color: var(--daa-blue);
  font-size: 11px;
  font-weight: 700;
}

@media (max-width: 720px) {
  .daa-route { grid-template-columns: 1fr 1fr; }
  .daa-card-head { display: block; }
  .daa-status { display: inline-block; margin-top: 7px; }
}

/* Full text is retained; wide evidence tables scroll rather than truncate. */
.daa-card, .daa-card-head > div, .daa-metric, .daa-result { min-width: 0; }
.daa-shell .daa-deck, .daa-shell .daa-summary, .daa-shell .daa-formula,
.daa-shell .daa-note, .daa-shell .daa-details, .daa-shell .daa-result p {
  white-space: normal; overflow-wrap: anywhere; word-break: normal;
  max-height: none; -webkit-line-clamp: unset; text-overflow: clip;
}
.daa-shell pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: none; }
.daa-metrics { grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr)); }
.daa-diagnostic-table { display: block; max-width: 100%; overflow-x: auto; table-layout: auto; }
.daa-diagnostic-table th, .daa-diagnostic-table td {
  min-width: 5.2rem; max-width: 24rem; white-space: normal;
  overflow-wrap: anywhere; word-break: normal; text-align: left !important;
}
.daa-diagnostic-table td.num, .daa-diagnostic-table th.num { text-align: left !important; }
.daa-gate-table { display: table; table-layout: fixed; width: 100%; }
.daa-gate-table th { width: 58%; font-size: 10px; text-transform: none; letter-spacing: 0; }
.daa-gate-table td { width: 42%; }
.daa-details td:first-child { width: 32%; }


/* Front-end cards share the publication stylesheet used by Stages 03--07. */
.daa-frontend .daa-front-table { display: table; width: 100%; table-layout: auto; }
.daa-front-scroll { max-width: 100%; overflow-x: auto; margin: 8px 0; }
.daa-frontend .daa-front-table th, .daa-frontend .daa-front-table td {
  min-width: 0; max-width: none; white-space: normal; overflow-wrap: anywhere;
  text-align: left !important; vertical-align: top;
}
.daa-frontend .daa-front-table th { font-size: 10px; letter-spacing: .03em; }
.daa-frontend .daa-front-variable { width: 12%; font-weight: 650; }
.daa-frontend .daa-front-si { font-family: Georgia, "Times New Roman", serif; white-space: normal; }
.daa-frontend .daa-front-si sup { font-size: .75em; line-height: 0; }
.daa-frontend .daa-front-matrix { table-layout: fixed; }
.daa-frontend .daa-front-matrix th, .daa-frontend .daa-front-matrix td { text-align: center !important; }
.daa-frontend .daa-front-matrix th:first-child { width: 100px; }
.daa-frontend .daa-result-grid { grid-template-columns: repeat(auto-fit, minmax(min(100%, 290px), 1fr)); }
.daa-frontend .daa-front-reason { margin-top: 5px; color: #475467; line-height: 1.45; }
.daa-frontend pre { margin: 9px 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; }
.daa-frontend details { margin-top: 10px; }
.daa-frontend .daa-front-caption { margin: 7px 0; color: #667085; font-size: 11px; }

</style>
"""


_REGIME_PLOT_COLORS = [
    "#299BD8", "#F2AF42", "#A78BD4", "#009E73",
    "#9b7ac9", "#4f9da6", "#b07d62", "#87946e",
]


def _inject_agent_editorial_styles() -> None:
    global _AGENT_EDITORIAL_STYLE_LOADED
    if not _AGENT_EDITORIAL_STYLE_LOADED:
        display(HTML(_AGENT_EDITORIAL_CSS))
        _AGENT_EDITORIAL_STYLE_LOADED = True


def _workflow_ui_enabled(hyperparameters: Optional[dict] = None) -> bool:
    hp = dict(hyperparameters or {})
    return bool(hp.get("show_workflow_progress", True))


def _ui_text(value: Any, max_length: int=220) -> str:
    if value is None:
        result = '—'
    elif isinstance(value, float):
        result = f'{value:.6g}'
    elif isinstance(value, (dict, list, tuple, np.ndarray)):
        result = json.dumps(value, ensure_ascii=False, default=str)
    else:
        result = str(value)
    result = result.replace('\n', ' · ').strip()
    return result


def _stage_identity(title: str) -> tuple[str, str]:
    raw = str(title).strip()
    exact = {
        "Dimensional Analysis Agent": ("Research workflow", "Dimensional Analysis Agent"),
        "Stage 1/7 · Problem and variable analysis": ("Stage 01", "Problem formulation"),
        "Stage 2/7 · Physics consistency audit": ("Stage 02", "Physical consistency"),
        "Stage 3/7 · Buckingham–Pi basis": ("Stage 03", "Dimensional basis"),
        "Stage 4/7 · Workflow planner": ("Stage 04", "Analysis protocol"),
        "Stage 5/7 · Mechanism screening": ("Stage 05A", "Physical-regime identification"),
        "Stage 5/7 · Numerical workflow execution": ("Stage 05B", "Local model selection"),
        "Stage 6/7 · Final summary": ("Stage 06", "Selected relations"),
        "Stage 7/7 · Data-collapse visualization": ("Stage 07", "Evidence figures"),
    }
    if raw in exact:
        return exact[raw]
    if raw.startswith("Local adaptive workflow"):
        return "Local analysis", raw.replace("Local adaptive workflow · ", "")
    if "candidate k=" in raw:
        return "Model evidence", raw
    if "count decision" in raw or "skipped k=" in raw or " failed" in raw:
        return "Model selection", raw
    if raw.endswith("· sparsification"):
        return "Parsimony check", raw.split(" · ")[0]
    if raw.endswith("· rationalization"):
        return "Formula refinement", raw.split(" · ")[0]
    if raw.endswith("· local workflow complete"):
        return "Local result", raw.split(" · ")[0]
    return "Workflow event", raw


def _status_style(status: str) -> tuple[str, str, str]:
    key = str(status).lower()
    mapping = {
        "running": ("info", "Running", "info"),
        "completed": ("success", "Complete", "success"),
        "passed": ("success", "Passed", "success"),
        "warning": ("warning", "Review", "warning"),
        "failed": ("danger", "Failed", "danger"),
        "skipped": ("muted", "Skipped", "info"),
        "info": ("muted", "Info", "info"),
    }
    return mapping.get(key, ("muted", str(status), "info"))


def _details_html(rows=None, bullets=None, code=None) -> str:
    pieces = ["<details class='daa-details'><summary>Methods and diagnostics</summary>"]
    if rows:
        pieces.append("<table>")
        for key, value in rows:
            pieces.append(
                "<tr><td>" + html.escape(_ui_text(key, 80)) + "</td><td>" +
                html.escape(_ui_text(value, 700)) + "</td></tr>"
            )
        pieces.append("</table>")
    if bullets:
        pieces.append("<ul>")
        pieces.extend("<li>" + html.escape(str(item)) + "</li>" for item in bullets)
        pieces.append("</ul>")
    if code:
        pieces.append("<div class='daa-formula'>" + html.escape(str(code).strip()) + "</div>")
    pieces.append("</details>")
    return "".join(pieces)


def display_workflow_stage(
    title: str,
    *,
    status: str = "running",
    summary: Optional[str] = None,
    rows: Optional[List[tuple[str, Any]]] = None,
    bullets: Optional[List[str]] = None,
    code: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> None:
    """Render the single fixed publication-style notebook card."""
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()
    kicker, clean_title = _stage_identity(title)
    tone, status_label, status_class = _status_style(status)

    all_rows = list(rows or [])
    visible_rows = all_rows[:5]
    hidden_rows = all_rows[5:]

    metric_html = ""
    if visible_rows:
        metric_html = "<div class='daa-metrics'>" + "".join(
            "<div class='daa-metric'><div class='daa-metric-key'>" +
            html.escape(_ui_text(key, 70)) + "</div><div class='daa-metric-value'>" +
            html.escape(_ui_text(value, 180)) + "</div></div>"
            for key, value in visible_rows
        ) + "</div>"

    code_lines = [line.strip() for line in str(code or "").splitlines() if line.strip()]
    formula_html = "".join(
        "<div class='daa-formula'>" + html.escape(_ui_text(line, 700)) + "</div>"
        for line in code_lines
    )

    clean_bullets = [
        item
        for item in (bullets or [])
        if item is not None and str(item).strip()
    ]

    note_html = ""
    if clean_bullets:
        note_html = (
            "<div class='daa-note'>"
            + html.escape(str(clean_bullets[0]))
            + "</div>"
        )

    detail_html = ""
    if hidden_rows or len(clean_bullets) > 1:
        detail_html = _details_html(
            hidden_rows,
            clean_bullets[1:],
            None,
        )

    card = f"""
    <div class='daa-shell'>
      <section class='daa-card' data-tone='{tone}'>
        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>{html.escape(kicker)}</div>
            <h3 class='daa-card-title'>{html.escape(clean_title)}</h3>
          </div>
          <span class='daa-status {status_class}'>{html.escape(status_label)}</span>
        </div>
        {f"<div class='daa-summary'>{html.escape(str(summary).strip())}</div>" if summary else ""}
        {metric_html}
        {formula_html}
        {note_html}
        {detail_html}
      </section>
    </div>
    """
    display(HTML(card))


def display_workflow_path(
    stages: List[str],
    *,
    current: Optional[str] = None,
    completed_through: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> None:
    if not _workflow_ui_enabled(hyperparameters) or not stages:
        return
    _inject_agent_editorial_styles()
    stage_list = list(stages)

    try:
        current_index = stage_list.index(current) if current is not None else -1
    except ValueError:
        current_index = -1
    try:
        completed_index = (
            stage_list.index(completed_through)
            if completed_through is not None
            else current_index - 1
            if current_index >= 0
            else -1
        )
    except ValueError:
        completed_index = -1

    cells = []
    for index, stage in enumerate(stage_list, start=1):
        raw = str(stage)
        clean = raw.replace("_", " ").replace("Buckingham-Pi", "Dimensional basis")
        zero_based_index = index - 1
        if zero_based_index <= completed_index:
            state_class = " complete"
        elif zero_based_index == current_index:
            state_class = " current"
        else:
            state_class = ""
        cells.append(
            f"<div class='daa-route-step{state_class}'>"
            f"<div class='daa-route-index'>{index:02d}</div>"
            f"<div class='daa-route-label'>{html.escape(clean.title())}</div></div>"
        )
    display(HTML("<div class='daa-shell'><div class='daa-route'>" + "".join(cells) + "</div></div>"))


def display_agent_run_header(
    problem_prompt: str,
    csv_path: str,
    n_rows: Optional[int],
    columns: List[str],
    hyperparameters: Optional[dict] = None,
) -> None:
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()
    hp = dict(hyperparameters or {})
    template = hp.get("workflow_template", "planner selected")
    abstract = _ui_text(problem_prompt, 540)
    html_block = f"""
    <div class='daa-shell'>
      <header class='daa-masthead'>
        <div class='daa-kicker'>Data-driven dimensional analysis</div>
        <h1 class='daa-title'>Dimensional Analysis Agent</h1>
        <div class='daa-deck'>{html.escape(abstract)}</div>
        <div class='daa-meta'>
          <span><strong>Samples</strong> {html.escape(_ui_text(n_rows))}</span>
          <span><strong>Variables</strong> {len(columns)}</span>
          <span><strong>Protocol</strong> {html.escape(str(template).replace('_', ' '))}</span>
          <span><strong>Version</strong> {html.escape(_config.AGENT_VERSION)}</span>
          <span><strong>Reproducibility</strong> manual parameters override planner suggestions</span>
        </div>
      </header>
    </div>
    """
    display(HTML(html_block))
    display_workflow_path(
        [
            "problem formulation", "physical consistency", "dimensional basis",
            "analysis protocol", "physical-regime identification", "local model selection",
            "selected relations", "evidence figures",
        ],
        current="problem formulation",
        hyperparameters=hyperparameters,
    )
    _frontend_flush_pending()


def display_basis_summary(input_vars, output_var, pi_formulas, output_resolution, hyperparameters=None) -> None:
    formulas = [f"{name} = {formula}" for name, formula in (pi_formulas or {}).items()]
    display_workflow_stage(
        "Stage 3/7 · Buckingham–Pi basis",
        status="completed",
        summary=(
            "A full-precision generating basis of the dimensional null space was constructed. "
            "The basis is not unique; familiar physical groups are products and ratios of these generators."
        ),
        rows=[
            ("Input variables", len(input_vars)),
            ("Output", output_var),
            ("Output inference", output_resolution.get("source")),
            ("Basis dimension", len(pi_formulas or {})),
        ],
        bullets=[
            "These are basis generators, not yet the final governing groups selected by MI/CMA-ES."
        ],
        code="\n".join(formulas),
        hyperparameters=hyperparameters,
    )


def display_compiled_plan(plan, compiled, provenance, hyperparameters=None) -> None:
    """Display the selected protocol and the output-scaling decision."""
    local = dict(compiled.get('local_workflow', {}))
    stages = list(compiled.get('stages', plan.get('stages', [])))
    rationale = str(plan.get('rationale', '—'))
    similarity_type = str(local.get('similarity_type', 'first_kind'))
    similarity_is_auto = similarity_type == 'auto'
    input_constraints = local.get('fixed_in_power') or {}
    output_constraints = local.get('fixed_out_power') or {}
    display_workflow_stage('Stage 4/7 · Workflow planner', status='completed', summary='The macro protocol was selected; the similarity class will be resolved numerically within each local regime.' if similarity_is_auto else 'The macro protocol and similarity class were selected. Count locking and all acceptance decisions remain numerical.', rows=[('Protocol', str(compiled.get('template', '')).replace('_', ' ')), ('Similarity', similarity_type.replace('_', ' ')), ('Output optimization', 'screening: fixed + joint' if similarity_is_auto else bool(local.get('optimize_output', False))), ('Similarity decision · relative MI threshold', local.get('similarity_screening_max_relative_mi_change') if similarity_is_auto else '—'), ('Input constraints', input_constraints if input_constraints else 'none'), ('Output constraints', output_constraints if output_constraints else 'none'), ('Scaling confidence', 'pending numerical screen' if similarity_is_auto else local.get('output_scaling_confidence', '—')), ('Manual overrides', len((provenance or {}).get('manual_overrides', {}))), ('Replanning budget', compiled.get('max_replans'))], bullets=[str(local.get('output_scaling_reason', '')), 'Auto screening ignores fixed input/output basis powers and uses k=1 with the default input-output cosine threshold. Final constraints are restored only after the similarity class is resolved.' if similarity_is_auto else '', rationale], hyperparameters=hyperparameters)
    display_workflow_path(stages, current=stages[0] if stages else None, hyperparameters=hyperparameters)


def display_similarity_screening_summary(screening: dict, hyperparameters=None) -> None:
    """Render the numerical Type-I/Type-II screening evidence for one scope."""
    if not _workflow_ui_enabled(hyperparameters):
        return
    status = str(screening.get('status', 'unknown'))
    scope_id = str(screening.get('scope_id', 'global'))
    if status != 'success':
        display_workflow_stage(f'Similarity-class screening · {scope_id}', status='warning', summary=screening.get('message', 'The similarity-class screen did not complete.'), rows=[('Status', status)], hyperparameters=hyperparameters)
        return
    selected = str(screening.get('selected_similarity_type', 'unknown'))
    mi_pass = bool(screening.get('mi_change_negligible', False))
    relative_mi_change = screening.get('relative_mi_loss_without_output_scaling')
    mi_threshold = screening.get('max_relative_mi_change')
    joint_summary = screening.get('joint_output_candidate', {}) or {}
    fixed_summary = screening.get('fixed_output_candidate', {}) or {}
    screening_formula_lines = []
    fixed_symbolic = list(fixed_summary.get('input_basis_coordinate_formulas') or [])
    fixed_expanded = list(fixed_summary.get('input_basis_expanded_formulas') or [])
    for i, symbolic in enumerate(fixed_symbolic):
        expanded = fixed_expanded[i] if i < len(fixed_expanded) else symbolic
        screening_formula_lines.append(f'Fixed-output optimizer basis coordinates PI_{i + 1}: {symbolic} = {expanded}')
    joint_symbolic = list(joint_summary.get('input_basis_coordinate_formulas') or [])
    joint_expanded = list(joint_summary.get('input_basis_expanded_formulas') or [])
    for i, symbolic in enumerate(joint_symbolic):
        expanded = joint_expanded[i] if i < len(joint_expanded) else symbolic
        screening_formula_lines.append(f'Joint-output optimizer basis coordinates PI_{i + 1}: {symbolic} = {expanded}')
    if joint_summary.get('output_basis_coordinate_formula'):
        screening_formula_lines.append(f"Joint-output optimizer output coordinates: {joint_summary.get('output_basis_coordinate_formula')} = {joint_summary.get('output_basis_expanded_formula')}")
    if joint_summary.get('output_scaling_formula'):
        screening_formula_lines.append(f"Screening output scale: {joint_summary.get('output_scaling_formula')}")
    criterion_text = f"MI-only decision: {('Type I' if mi_pass else 'Type II')}"
    display_workflow_stage(f'Similarity-class screening · {scope_id}', status='completed', summary='The similarity class is determined only by the relative mutual-information loss incurred when output scaling is removed.', rows=[('Decision', selected.replace('_', ' ')), ('Screening k', screening.get('screening_k', _config.SIMILARITY_SCREEN_K)), ('MI · fixed output', screening.get('fixed_output_mutual_information')), ('MI · joint output', screening.get('joint_output_mutual_information')), ('Relative MI loss without output scaling', f'{100.0 * float(relative_mi_change):.3g}%' if relative_mi_change is not None else '—'), ('Relative MI threshold', f'≤ {100.0 * float(mi_threshold):.3g}%' if mi_threshold is not None else '—'), ('Criteria', criterion_text), ('Screening cosine threshold', screening.get('screening_max_cosine_similarity')), ('Fixed basis powers during screening', 'none'), ('Final input constraints', screening.get('final_fixed_in_power') or 'none'), ('Final output constraints', screening.get('final_fixed_out_power') or 'none')], bullets=[screening.get('reason', ''), 'Screening is diagnostic only: the selected local workflow is rerun from scratch and restores the user-specified basis constraints for final inference.', 'Type II was selected without an output-side uniqueness constraint; the final optimization will therefore retain the existing unconstrained Type-II behavior.' if selected == 'second_kind' and (not screening.get('final_fixed_out_power')) else ''], code='\n'.join(screening_formula_lines) if screening_formula_lines else None, hyperparameters=hyperparameters)


def _regime_diag_num(value, digits=5) -> str:
    """Compact numerical formatter for regime-screening diagnostics."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except Exception:
        return html.escape(str(value))
    if np.isnan(number):
        return "—"
    if np.isposinf(number):
        return "∞"
    if np.isneginf(number):
        return "−∞"
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1.0e4 or magnitude < 1.0e-3):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def _regime_diag_badge(flag, *, true_text="pass", false_text="reject") -> str:
    if flag is None:
        return "<span class='daa-mini-badge neutral'>—</span>"
    css_class = "pass" if bool(flag) else "fail"
    label = true_text if bool(flag) else false_text
    return (
        f"<span class='daa-mini-badge {css_class}'>"
        + html.escape(str(label))
        + "</span>"
    )


def _mechanism_gate_html(clustering):
    rows = [
        ('Candidate K', clustering.get('candidate_cluster_counts')),
        ('Fixed K', clustering.get('n_clusters')),
        ('Minimum cluster fraction', clustering.get('min_cluster_fraction')),
        ('Minimum cluster samples', clustering.get('min_cluster_samples')),
        ('Minimum eigenvalue ratio', clustering.get('eigenvalue_ratio_threshold')),
        ('Require convergence', clustering.get('require_convergence')),
        ('Auto K=2 directional support: |cos| <', clustering.get('max_intercluster_cosine')),
        ('Global spectral fraction', 'Diagnostic only; no decision threshold'),
    ]
    return "<details class='daa-regime-details' open><summary>Configured evidence gates</summary><table class='daa-diagnostic-table daa-gate-table'><tbody>" + ''.join(
        '<tr><th scope="row">' + html.escape(str(key)) + '</th><td>' + html.escape(_ui_text(value)) + '</td></tr>' for key, value in rows
    ) + '</tbody></table></details>'


def _display_mechanism_running(clustering, template, hyperparameters=None):
    if not _workflow_ui_enabled(hyperparameters):
        return None
    _inject_agent_editorial_styles()
    summary = ('Collecting all requested partition evidence and checking K=2 directional support.'
               if template == 'auto_mechanism_adaptive' else 'Collecting the explicitly requested partition evidence.')
    card = "<div class='daa-shell'><section class='daa-card' data-tone='info'><div class='daa-card-head'><div><div class='daa-card-label'>Stage 05A</div><h3 class='daa-card-title'>Physical-regime identification</h3></div><span class='daa-status info'>Running</span></div><div class='daa-summary'>" + html.escape(summary) + '</div>' + _mechanism_gate_html(clustering) + '</section></div>'
    return display(HTML(card), display_id=True)


def display_regime_detection_summary(detection: dict, hyperparameters=None, display_handle=None) -> None:
    """Render detailed GPR-gradient regime evidence without raw console dumps.

    This is display-only. It does not alter clustering, acceptance thresholds,
    selected K, labels, or downstream local workflows.
    """
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()
    hp = dict(hyperparameters or {})
    status = str(detection.get('status', 'unknown'))
    single = detection.get('single_regime_baseline') or {}
    surrogate = detection.get('surrogate') or detection.get('gpr') or {}
    candidates = list(detection.get('candidate_summary', []) or [])
    selected_clusters = list(detection.get('cluster_summary', []) or [])
    criteria = dict(detection.get('acceptance_criteria') or {})
    nested_cluster_hp = (hp.get('gradient_regime_options') or {}).get('clustering', {}) if isinstance(hp.get('gradient_regime_options') or {}, dict) else {}
    fraction_threshold = float(criteria.get('minimum_cluster_fraction', nested_cluster_hp.get('min_cluster_fraction', 0.05)))
    ratio_threshold = float(criteria.get('minimum_eigenvalue_ratio', nested_cluster_hp.get('eigenvalue_ratio_threshold', 50.0)))
    require_convergence = bool(criteria.get('require_convergence', nested_cluster_hp.get('require_convergence', True)))
    if status == 'not_requested':
        headline = detection.get('message', 'Prescribed single-regime analysis; gradient diagnostics not computed')
        tone = 'success'
        status_label = 'Not requested'
    elif status == 'global_analysis_selected':
        headline = 'Global analysis selected after collecting regime evidence'
        tone = 'success'
        status_label = 'Complete'
    elif status == 'success':
        headline = 'A multi-regime partition passed the evidence gate'
        tone = 'success'
        status_label = 'Complete'
    else:
        headline = detection.get('message', 'No admissible multi-regime partition was found')
        tone = 'warning'
        status_label = 'Review'
    selected_k = detection.get('best_k', 1 if status == 'global_analysis_selected' else None)
    usable_gradients = len(detection.get('valid_indices', []) or [])
    excluded_gradients = detection.get('excluded_directionless_samples', len(detection.get('excluded_directionless_indices', []) or []))
    configured = {'candidate_cluster_counts': detection.get('requested_cluster_counts', []), 'n_clusters': detection.get('requested_fixed_k'), 'min_cluster_fraction': fraction_threshold, 'min_cluster_samples': criteria.get('minimum_cluster_samples', 'not recorded'), 'eigenvalue_ratio_threshold': ratio_threshold, 'require_convergence': require_convergence, 'max_intercluster_cosine': criteria.get('maximum_intercluster_cosine', 'not recorded')}
    configuration_html = _mechanism_gate_html(configured)
    metric_items = [('Selected K', selected_k), ('Requested K', detection.get('requested_cluster_counts', 'not computed' if status == 'not_requested' else [])), ('Admissible K', (detection.get('decision_evidence') or {}).get('admissible_cluster_counts')), ('Failed K', [failure.get('k') for failure in detection.get('clustering_failures', [])]), ('Usable gradients', usable_gradients), ('Excluded gradients', excluded_gradients)]
    metric_html = "<div class='daa-metrics'>" + ''.join(("<div class='daa-metric'><div class='daa-metric-key'>" + html.escape(str(key)) + "</div><div class='daa-metric-value'>" + html.escape(_ui_text(value, 180)) + '</div></div>' for key, value in metric_items)) + '</div>'
    gpr_rows = [('Mean mode', surrogate.get('mean_mode')), ('Training samples', surrogate.get('n_train_samples')), ('sigma0²', surrogate.get('sigma0_squared')), ('sigma1²', surrogate.get('sigma1_squared')), ('sigma_u²', surrogate.get('sigma_u_squared')), ('Negative log likelihood', surrogate.get('negative_log_likelihood')), ('Optimizer success', surrogate.get('optimizer_success')), ('Optimizer iterations', surrogate.get('optimizer_iterations'))]
    gpr_table = "<div class='daa-section-title'>GPR surrogate</div><table class='daa-diagnostic-table'><tbody>" + ''.join(('<tr><td>' + html.escape(str(key)) + "</td><td class='num'>" + html.escape(str(value) if isinstance(value, (bool, np.bool_)) else _regime_diag_num(value) if isinstance(value, (int, float, np.number)) else _ui_text(value, 260)) + '</td></tr>' for key, value in gpr_rows)) + '</tbody></table>'
    restart_rows = []
    for item in surrogate.get('optimizer_restarts', []) or []:
        restart_rows.append(f"<tr><td>{html.escape(str(int(item.get('restart_id', 0)) + 1))}</td><td class='num'>{html.escape(_regime_diag_num(item.get('negative_log_likelihood')))}</td><td>{_regime_diag_badge(item.get('success'), true_text='success', false_text='stopped')}</td><td class='num'>{html.escape(_regime_diag_num(item.get('iterations'), digits=6))}</td><td class='num'>{html.escape(_ui_text(item.get('message'), 260))}</td></tr>")
    restart_html = ''
    if restart_rows:
        restart_html = "<div class='daa-section-title'>Optimizer restarts</div><table class='daa-diagnostic-table'><thead><tr><th>Restart</th><th class='num'>NLL</th><th>Status</th><th class='num'>Iterations</th><th class='num'>Messages</th></tr></thead><tbody>" + ''.join(restart_rows) + '</tbody></table>'
    gpr_details_html = "<details class='daa-regime-details'><summary>GPR surrogate and optimizer diagnostics</summary>" + gpr_table + restart_html + '</details>'
    global_html = ''
    if single:
        eigenvalues = single.get('eigenvalues', []) or []
        global_html = "<div class='daa-section-title'>Global normalized-gradient covariance</div><table class='daa-diagnostic-table'><tbody><tr><td>Eigenvalues</td><td>" + html.escape('[' + ', '.join((_regime_diag_num(v) for v in eigenvalues)) + ']') + "</td></tr><tr><td>λ1 / Σλ</td><td class='num'>" + html.escape(_regime_diag_num(single.get('leading_fraction'))) + '</td></tr><tr><td>Global spectrum role</td><td>' + html.escape(str(single.get('role', 'diagnostic_only_no_decision_threshold'))) + '</td></tr></tbody></table>'
    candidate_html = ''
    candidate_detail_html = ''
    if candidates:
        candidate_rows = []
        details = []
        for candidate in sorted(candidates, key=lambda item: int(item.get('k', 0))):
            k = int(candidate.get('k', 0))
            admissible = bool(candidate.get('meets_regime_criteria', False)) and (bool(candidate.get('converged', False)) or not require_convergence)
            selected_mark = ' · selected' if selected_k is not None and int(selected_k) == k and (status == 'success') else ''
            candidate_rows.append(f"<tr><td><strong>K={k}</strong>{html.escape(selected_mark)}</td><td>{_regime_diag_badge(candidate.get('converged'), true_text='yes', false_text='no')}</td><td class='num'>{float(candidate.get('minimum_cluster_fraction', 0.0)):.1%}</td><td>{_regime_diag_badge(candidate.get('passes_fraction_criterion'))}</td><td class='num'>{html.escape(_ui_text(candidate.get('minimum_cluster_samples'), 60))}</td><td>{_regime_diag_badge(candidate.get('passes_sample_count_criterion'))}</td><td class='num'>{html.escape(_regime_diag_num(candidate.get('minimum_eigenvalue_ratio')))}</td><td>{_regime_diag_badge(candidate.get('passes_eigenvalue_ratio_criterion'))}</td><td class='num'>{html.escape(_regime_diag_num(candidate.get('max_intercluster_abs_cosine')))}</td><td class='num'>{html.escape(_regime_diag_num(candidate.get('minimum_intercluster_angle_degrees')))}</td><td>{html.escape(str(candidate.get('k2_direction_action', 'not recorded')).replace('_', ' '))}</td><td>{_regime_diag_badge(admissible, true_text='admissible', false_text='reject')}</td><td class='num'>{html.escape(_regime_diag_num(candidate.get('clustering_objective')))}</td></tr>")
            cluster_rows = []
            for cluster in candidate.get('clusters', []) or []:
                regime_id = int(cluster.get('regime_id', cluster.get('cluster', 0))) + 1
                eigenvalues = cluster.get('eigenvalues', []) or []
                cluster_rows.append(f"<tr><td>R{regime_id}</td><td class='num'>{html.escape(_ui_text(cluster.get('n_samples'), 80))}</td><td class='num'>{float(cluster.get('fraction', 0.0)):.1%}</td><td class='num'>{html.escape(_regime_diag_num(cluster.get('eigenvalue_ratio_max_to_min')))}</td><td>{html.escape('[' + ', '.join((_regime_diag_num(v) for v in eigenvalues)) + ']')}</td></tr>")
            direction_pairs = (candidate.get('direction_separation') or {}).get('pairs', [])
            direction_pair_html = ''
            if direction_pairs:
                direction_pair_html = "<table class='daa-diagnostic-table'><thead><tr><th>Cluster pair</th><th>|cos θ|</th><th>Angle (deg)</th></tr></thead><tbody>" + ''.join((f"<tr><td>R{pair['cluster_i'] + 1}–R{pair['cluster_j'] + 1}</td><td>{html.escape(_regime_diag_num(pair['absolute_cosine']))}</td><td>{html.escape(_regime_diag_num(pair['angle_degrees']))}</td></tr>" for pair in direction_pairs)) + '</tbody></table>'
            details.append(f"<details class='daa-regime-details' open><summary>K={k} cluster composition and eigenspectra</summary><table class='daa-diagnostic-table'><thead><tr><th>Regime</th><th class='num'>Samples</th><th class='num'>Fraction</th><th class='num'>λmax / λmin</th><th>Eigenvalues</th></tr></thead><tbody>" + (''.join(cluster_rows) if cluster_rows else "<tr><td colspan='5'>No cluster detail recorded.</td></tr>") + '</tbody></table>' + direction_pair_html + "<div class='daa-diagnostic-note'>" + html.escape('Termination: ' + str(candidate.get('termination_reason', '—')) + f" · iterations={candidate.get('iterations', '—')}") + '</div></details>')
        candidate_html = "<div class='daa-section-title'>Candidate partition evidence</div><table class='daa-diagnostic-table'><thead><tr><th>K</th><th>Converged</th><th class='num'>Min fraction</th><th>Fraction gate</th><th class='num'>Min samples</th><th>Sample-count gate</th><th class='num'>Min eigenvalue ratio</th><th>Ratio gate</th><th class='num'>K=2 |cos|</th><th class='num'>K=2 angle (deg)</th><th>K=2 action</th><th>Base admissibility</th><th class='num'>Objective</th></tr></thead><tbody>" + ''.join(candidate_rows) + '</tbody></table>'
        candidate_detail_html = ''.join(details)
    failure_rows = []
    for failure in detection.get('clustering_failures', []) or []:
        failure_rows.append('<tr><td>K=' + html.escape(str(failure.get('k', '—'))) + '</td><td>' + html.escape(str(failure.get('status') or 'No valid candidate partition returned')) + '</td></tr>')
    failure_html = "<div class='daa-section-title'>Requested K without usable partition evidence</div><table class='daa-diagnostic-table'><thead><tr><th>K</th><th>Status</th></tr></thead><tbody>" + ''.join(failure_rows) + '</tbody></table>' if failure_rows else ''
    card = f"\n    <div class='daa-shell'>\n      <section class='daa-card' data-tone='{tone}'>\n        <div class='daa-card-head'>\n          <div>\n            <div class='daa-card-label'>Stage 05A</div>\n            <h3 class='daa-card-title'>Physical-regime identification</h3>\n          </div>\n          <span class='daa-status {('success' if tone == 'success' else 'warning')}'>{html.escape(status_label)}</span>\n        </div>\n        <div class='daa-summary'>{html.escape(str(headline))}</div>\n        {configuration_html}\n{metric_html}\n        {gpr_details_html}\n        {global_html}\n        {candidate_html}\n        {failure_html}\n        {candidate_detail_html}\n      </section>\n    </div>\n    "
    if display_handle is not None and hasattr(display_handle, 'update'):
        display_handle.update(HTML(card))
    else:
        display(HTML(card))
    progress_stages = list(hp.get('_workflow_progress_stages', []) or [])
    if progress_stages:
        progress_current = 'validate_regime_partition' if 'validate_regime_partition' in progress_stages else 'gradient_regime_detection' if 'gradient_regime_detection' in progress_stages else progress_stages[0]
        display_workflow_path(progress_stages, current=progress_current, hyperparameters=hp)


def _candidate_uncertainty_text(candidate: dict) -> str:
    uncertainty = _plotting._candidate_uncertainty(candidate)
    return (
        f"[ MI = {uncertainty['combined_mi_std']:.3g} Res={uncertainty['restart_mi_std']:.3g}, Sub={uncertainty['subsample_mi_std']:.3g} ]"
        f"[ Score = {uncertainty['combined_score_std']:.3g} Res={uncertainty['restart_score_std']:.3g}, Sub={uncertainty['subsample_score_std']:.3g} ]"
    )


def display_local_candidate(
    scope_id: str,
    candidate: dict,
    hyperparameters=None,
) -> None:
    validation = candidate.get("gpr_validation") or {}
    stability = candidate.get("restart_stability") or {}
    formula_lines = _optimization._candidate_formula_lines(
        candidate,
        output_label=str(candidate.get("output_label", "Y")),
    )
    display_workflow_stage(
        f"{scope_id} · candidate k={candidate.get('pi_count')}",
        status="completed",
        summary=(
            "A candidate representation was optimized and checked "
            "against restart and held-out evidence."
        ),
        rows=[
            ("Mutual information", candidate.get("mutual_information")),
            (
                "Normalized information score ε",
                candidate.get("normalized_information_score"),
            ),
            ("Uncertainty", _candidate_uncertainty_text(candidate)),
            ("Held-out R²", validation.get("test_r2_mean")),
            ("Successful restarts", candidate.get("restart_count")),            
            (
                "Restart stability",
                stability.get("minimum_principal_cosine"),
            ),
        ],
        bullets=[
            "Physical exponents are display-normalized; basis-coordinate powers are the raw optimizer coefficients."
        ],
        code="\n".join(formula_lines),
        hyperparameters=hyperparameters,
    )


def display_count_decision(
    scope_id: str,
    decision: dict,
    hyperparameters=None,
) -> None:
    """Display only count decisions that affect the executed path."""
    event = decision.get("event")

    if event == "COUNT_TEST_SKIPPED_INSUFFICIENT_SAMPLES":
        display_workflow_stage(
            f"{scope_id} · skipped k={decision.get('pi_count')}",
            status="warning",
            summary=(
                "The candidate was not fitted because the reproducible "
                "sample-size guard was not met."
            ),
            rows=[
                ("Available samples", decision.get("n_samples")),
                ("Required samples", decision.get("minimum_required")),
                ("Guard", decision.get("sample_guard")),
            ],
            hyperparameters=hyperparameters,
        )
    elif event == "COUNT_OPTIMIZATION_FAILED":
        display_workflow_stage(
            f"{scope_id} · k={decision.get('pi_count')} failed",
            status="warning",
            summary=(
                "All CMA-ES restarts failed; the last successful "
                "dimensionality is retained."
            ),
            rows=[("Diagnostic", decision.get("details"))],
            hyperparameters=hyperparameters,
        )


def display_local_final(
    local_result: dict,
    hyperparameters=None,
) -> None:
    if not _workflow_ui_enabled(hyperparameters):
        return

    _inject_agent_editorial_styles()

    final = local_result.get("final_solution", {}) or {}
    gpr = final.get("gpr_validation", {}) or {}
    sparsification = local_result.get("sparsification", {}) or {}
    rationalization = local_result.get("rationalization", {}) or {}

    formula_lines = _optimization._candidate_formula_lines(
        final,
        output_label=str(final.get("output_label", "Y")),
    )

    # --------------------------------------------------------------
    # Count-selection evidence
    # --------------------------------------------------------------
    decisions = list(local_result.get("count_decisions", []) or [])

    count_comparison = next(
        (
            item for item in reversed(decisions)
            if item.get("event") == "COUNT_COMPARISON"
        ),
        None,
    )

    if count_comparison:
        count_evidence = (
            f"k={count_comparison.get('previous_count')} vs "
            f"k={count_comparison.get('current_count')}; "
            f"Δscore={_ui_text(count_comparison.get('absolute_score_drop'))}; "
            f"required ≥ {_ui_text(count_comparison.get('required_drop'))}; "
            f"beyond uncertainty="
            f"{_ui_text(count_comparison.get('exceeds_uq_margin'))}"
        )
    else:
        count_evidence = (
            f"tested={_ui_text(local_result.get('tested_counts'))}; "
            f"{_ui_text(local_result.get('lock_reason'))}"
        )

    # --------------------------------------------------------------
    # Sparsification evidence
    # --------------------------------------------------------------
    sparse_parts = []

    if "dropped_variables" in sparsification:
        sparse_parts.append(
            "dropped=" + _ui_text(
                sparsification.get("dropped_variables", [])
            )
        )

    if sparsification.get("score_increase") is not None:
        sparse_parts.append(
            "Δscore="
            + _ui_text(sparsification.get("score_increase"))
        )

    if sparsification.get("maximum_allowed_score_increase") is not None:
        sparse_parts.append(
            "allowed ≤ "
            + _ui_text(
                sparsification.get("maximum_allowed_score_increase")
            )
        )

    sparse_evidence = "; ".join(sparse_parts) or "—"

    # --------------------------------------------------------------
    # Rationalization evidence
    # --------------------------------------------------------------
    rational_status = (
        "accepted"
        if rationalization.get("accepted")
        else rationalization.get("status", "not adopted")
    )

    rational_parts = []

    if rationalization.get("relative_mi_drop") is not None:
        rational_parts.append(
            "relative MI loss="
            + _ui_text(rationalization.get("relative_mi_drop"))
        )

    if rationalization.get("max_relative_mi_drop") is not None:
        rational_parts.append(
            "allowed ≤ "
            + _ui_text(rationalization.get("max_relative_mi_drop"))
        )

    rational_evidence = "; ".join(rational_parts) or "—"

    # --------------------------------------------------------------
    # Final evidence
    # --------------------------------------------------------------
    final_evidence = (
        f"MI={_ui_text(final.get('mutual_information'))}; "
        f"residual={_ui_text(final.get('normalized_information_score'))}; "
        f"held-out R²="
        f"{_ui_text(gpr.get('test_r2_mean', final.get('gpr_test_r2')))}"
    )

    rows = [
        (
            "Count selection",
            f"k={_ui_text(local_result.get('locked_count'))}",
            count_evidence,
        ),
        (
            "Sparsification",
            _ui_text(sparsification.get("status")),
            sparse_evidence,
        ),
        (
            "Rationalization",
            _ui_text(rational_status),
            rational_evidence,
        ),
        (
            "Final model",
            "complete"
            if local_result.get("status") == "success"
            else "review",
            final_evidence,
        ),
    ]

    table_html = (
        "<table class='daa-diagnostic-table'>"
        "<thead><tr>"
        "<th>Step</th>"
        "<th>Result</th>"
        "<th>Evidence</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td><strong>{html.escape(str(step))}</strong></td>"
            f"<td>{html.escape(str(result))}</td>"
            f"<td>{html.escape(str(evidence))}</td>"
            "</tr>"
            for step, result, evidence in rows
        )
        + "</tbody></table>"
    )

    formula_html = "".join(
        "<div class='daa-formula'>"
        + html.escape(_ui_text(line, 700))
        + "</div>"
        for line in formula_lines
    )

    success = local_result.get("status") == "success"

    card = f"""
    <div class='daa-shell'>
      <section class='daa-card'
               data-tone='{"success" if success else "warning"}'>

        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>Local result</div>
            <h3 class='daa-card-title'>
              {html.escape(str(local_result.get('scope_id')))}
            </h3>
          </div>

          <span class='daa-status {"success" if success else "warning"}'>
            {"Complete" if success else "Review"}
          </span>
        </div>

        <div class='daa-summary'>
          Count selection, parsimony checks, and formula refinement
          are summarized below.
        </div>

        {table_html}

        {formula_html}

        <div class='daa-note'>
          The physical and basis-coordinate forms are
          scale-equivalent representations of the selected group.
        </div>

      </section>
    </div>
    """

    display(HTML(card))

    progress_stages = list(dict(hyperparameters or {}).get("_workflow_progress_stages", []) or [])
    if progress_stages:
        progress_current = (
            "local_dominant_group_search"
            if "local_dominant_group_search" in progress_stages
            else "fan_out_local_workflows"
            if "fan_out_local_workflows" in progress_stages
            else progress_stages[-1]
        )
        display_workflow_path(
            progress_stages,
            current=progress_current,
            hyperparameters=hyperparameters,
        )


def display_final_workflow_summary(
    state: dict,
    deterministic_report: str,
    report_text: str,
) -> None:
    """Display the final deterministic results with paired formula representations."""
    hp = dict(state.get("manual_hyperparams", {}) or {})
    result = state.get("workflow_result", {}) or {}
    local_results = result.get("local_workflows", []) or []
    output_label = str(state.get("output_var") or "Y")

    _inject_agent_editorial_styles()
    cards = []
    for local in local_results:
        final = local.get("final_solution", {}) or {}
        gpr = final.get("gpr_validation", {}) or {}
        formula_lines = _optimization._candidate_formula_lines(
            final,
            output_label=output_label,
        )
        formula_html = "".join(
            "<div class='daa-formula'>" +
            html.escape(_ui_text(line, 900)) +
            "</div>"
            for line in formula_lines
        )

        requested_similarity = str(
            local.get(
                "similarity_type_requested",
                local.get("similarity_type", "first_kind"),
            )
        )

        resolved_similarity = str(
            local.get("similarity_type", "first_kind")
        )

        requested_display = requested_similarity.replace("_", " ")
        resolved_display = resolved_similarity.replace("_", " ")

        if requested_similarity == "auto":
            similarity_display = f"auto → {resolved_display}"
        else:
            similarity_display = f"set → {requested_display}"

        cards.append(
            "<article class='daa-result'>"
            f"<h4>{html.escape(str(local.get('scope_id', 'global')).replace('_', ' ').title())}</h4>"
            f"<p><strong>Selected k:</strong> {html.escape(_ui_text(local.get('locked_count')))}</p>"
            f"<p><strong>Similarity:</strong> {html.escape(_ui_text(similarity_display))}</p>"            
            f"<p><strong>Output optimization:</strong> {html.escape(_ui_text(local.get('optimize_output', False)))}</p>"
            f"<p><strong>MI / information score:</strong> "
            f"{html.escape(_ui_text(final.get('mutual_information')))} / "
            f"{html.escape(_ui_text(final.get('normalized_information_score')))}</p>"
            f"<p><strong>Held-out R²:</strong> "
            f"{html.escape(_ui_text(gpr.get('test_r2_mean', final.get('gpr_test_r2'))))}</p>"
            f"{formula_html or '<div class=\"daa-formula\">No formula</div>'}"
            "</article>"
        )

    block = f"""
    <div class='daa-shell'>
      <section class='daa-card' data-tone='success'>
        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>Stage 06</div>
            <h3 class='daa-card-title'>Selected dimensionless relations</h3>
          </div>
          <span class='daa-status success'>
            {html.escape(str(result.get('status', 'complete')).replace('_', ' '))}
          </span>
        </div>
        <div class='daa-summary'>
          Final relations are copied directly from the numerical state.
          Each relation is shown both in physical variables and in the
          optimized Buckingham-Pi basis coordinates.
        </div>
        <div class='daa-metrics'>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Output</div>
            <div class='daa-metric-value'>{html.escape(output_label)}</div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Mechanism decision</div>
            <div class='daa-metric-value'>
              {html.escape(_ui_text(result.get('mechanism_decision')).replace('_', ' '))}
            </div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Protocol</div>
            <div class='daa-metric-value'>
              {html.escape(_ui_text(result.get('executed_template')).replace('_', ' '))}
            </div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Local models</div>
            <div class='daa-metric-value'>{len(local_results)}</div>
          </div>
        </div>
        <div class='daa-result-grid'>{''.join(cards)}</div>
      </section>
    </div>
    """
    display(HTML(block))
    progress_stages = list((state.get("compiled_workflow") or {}).get("stages", []) or [])
    if progress_stages:
        # Aggregation is finished here. Keep Generate Report inactive until
        # the final notebook output (including evidence figures) has completed.
        display_workflow_path(
            progress_stages,
            completed_through=(
                "aggregate_results"
                if "aggregate_results" in progress_stages
                else progress_stages[-2]
                if len(progress_stages) > 1
                else None
            ),
            hyperparameters=hp,
        )

    if (
        hp.get("show_llm_report_in_notebook", True)
        and str(report_text).strip()
        and str(report_text).strip() != str(deterministic_report).strip()
    ):
        interpretation_kb = state.get("interpretation_knowledge_retrieval", {}) or {}
        grounding_note = ""
        rag_evidence_html = ""
        if interpretation_kb.get("status") == "success":
            grounding_note = (
                "<div class='daa-diagnostic-note'>"
                "Knowledge grounding · documented dimensionless relations · "
                + html.escape(_ui_text(interpretation_kb.get("embedding_model"), 80))
                + " · retrieved "
                + html.escape(_ui_text(interpretation_kb.get("n_retrieved_records", 0), 40))
                + " reference records</div>"
            )

            rag_matches = list(interpretation_kb.get("matches", []) or [])
            if rag_matches:
                rag_rows = []
                for match in rag_matches:
                    rag_rows.append(
                    "<tr>"
                    "<td><strong>"
                    + html.escape(str(match.get("canonical_name", "—")))
                    + "</strong></td>"
                    "<td>"
                    + html.escape(_ui_text(match.get("domain", "—"), 100))
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('retrieval_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('lexical_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('semantic_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td>"
                    + html.escape(
                        str(match.get("retrieval_mode", "—")).replace("_", " ")
                    )
                    + "</td>"
                    "</tr>"
                    )

                rag_evidence_html = (
                    "<details class='daa-details'>"
                    "<summary>"
                    "Retrieved dimensionless knowledge · top 3 matches"
                    "</summary>"

                    "<table class='daa-diagnostic-table'>"

                    "<thead>"
                    "<tr>"
                    "<th>Relation</th>"
                    "<th>Domain</th>"
                    "<th class='num'>Score</th>"
                    "<th class='num'>Lexical</th>"
                    "<th class='num'>Semantic</th>"
                    "<th>Mode</th>"
                    "</tr>"
                    "</thead>"

                    "<tbody>"
                    + "".join(rag_rows)
                    + "</tbody>"

                    "</table>"
                    "</details>"
                )

        display(HTML(
            "<div class='daa-shell'><section class='daa-card' data-tone='muted'>"
            "<div class='daa-card-label'>Model output</div>"
            f"<div class='daa-summary'>{html.escape(_ui_text(report_text, 2000))}</div>"
            f"{grounding_note}"
            f"{rag_evidence_html}"
            "</section></div>"
        ))


def save_State_2md(state: dict, save_name: str, if_save=False):
    """Save one compact, human-readable Markdown report instead of dumping the full state."""
    case_name = save_name.split('_')[0].upper()
    wk_space = str(_config.OUTPUT_DIR / case_name)
    os.makedirs(wk_space, exist_ok=True)
    save_path = os.path.join(wk_space, f'{save_name}.md')
    if not if_save:
        print(f'⚠️ To path {save_path} State save NOT ON.')
        return save_path

    def md_text(value, max_length=1500):
        if value is None or value == '':
            return '—'
        if isinstance(value, float):
            return f'{value:.6g}'
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        value = str(value).replace('\r', '').strip()
        return value

    def md_cell(value, max_length=240):
        return md_text(value, max_length).replace('|', '\\|').replace('\n', '<br>')

    def write_table(handle, rows):
        handle.write('| Item | Value |\n|---|---|\n')
        for key, value in rows:
            handle.write(f'| {md_cell(key, 80)} | {md_cell(value)} |\n')
        handle.write('\n')

    def md_pct(value):
        if value is None or value == '':
            return '—'
        try:
            return f'{float(value):.1%}'
        except Exception:
            return md_text(value)

    def md_vec(value, digits=6):
        if value is None or value == '':
            return '—'
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return md_text(value, 2000)
        if arr.ndim == 0:
            return f'{float(arr):.{digits}g}'
        if arr.ndim == 1:
            return '[' + ', '.join((f'{float(x):.{digits}g}' for x in arr.tolist())) + ']'
        return json.dumps(arr.tolist(), ensure_ascii=False)

    def write_wide_table(handle, headers, rows):
        handle.write('| ' + ' | '.join((md_cell(h, 120) for h in headers)) + ' |\n')
        handle.write('|' + '|'.join(['---'] * len(headers)) + '|\n')
        for row in rows:
            handle.write('| ' + ' | '.join((md_cell(v, 1200) for v in row)) + ' |\n')
        handle.write('\n')

    def write_rag_evidence_block(handle, summary: str, evidence: Any) -> None:
        text = str(evidence or '').strip()
        if not text:
            return
        handle.write('<details>\n')
        handle.write(f'<summary>{md_cell(summary, 160)}</summary>\n\n')
        handle.write(text + '\n\n')
        handle.write('</details>\n\n')
    result = state.get('workflow_result', {}) or {}
    compiled = state.get('compiled_workflow', {}) or {}
    local_options = compiled.get('local_workflow', {}) or {}
    local_workflows = result.get('local_workflows', []) or []
    output_var = state.get('output_var') or state.get('analyzed_output_var') or '—'
    input_vars = state.get('input_vars') or state.get('analyzed_input_vars') or []
    regime_detection = result.get('regime_detection', {}) or {}
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('# Dimensional Analysis Agent Report\n\n')
        write_table(f, [('Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ('Status', result.get('status', state.get('workflow_status', '—'))), ('Data', state.get('csv_FilePath', '—')), ('Agent version', _config.AGENT_VERSION)])
        f.write('## 1. Problem\n\n')
        f.write(md_text(state.get('prompt_problem', '—'), 5000) + '\n\n')
        write_table(f, [('Dependent variable', output_var), ('Independent inputs', input_vars), ('Output inference', (state.get('output_resolution') or {}).get('source', state.get('analyzed_output_reasoning', '—')))])
        try:
            _write_frontend_markdown(f, state)
        except Exception as display_error:
            f.write("Front-end report formatting warning: " + str(display_error) + "\n\n")
        basis = state.get('basis_pi_formulas', {}) or {}
        if basis:
            f.write('### Buckingham–Pi basis\n\n')
            f.write('| Group | Formula |\n|---|---|\n')
            for name, formula in basis.items():
                f.write(f'| {md_cell(name, 80)} | `{md_cell(formula, 500)}` |\n')
            f.write('\n')
        f.write('## 3. Analysis protocol\n\n')
        write_table(f, [('Protocol', str(compiled.get('template', '—')).replace('_', ' ')), ('Similarity', str(local_options.get('similarity_type', result.get('similarity_type', '—'))).replace('_', ' ')), ('Output optimization', local_options.get('optimize_output', result.get('output_optimization', False))), ('Input constraints', local_options.get('fixed_in_power') or 'none'), ('Output constraints', local_options.get('fixed_out_power') or 'none'), ('CMA-ES restarts', local_options.get('cma_restarts', '—')), ('Mechanism decision', str(result.get('mechanism_decision', '—')).replace('_', ' '))])
        preprocessing = result.get('dimensionless_input_preprocessing', {})
        if preprocessing:
            write_table(f, [('Dimensionless input preprocessing', _workflow._format_preprocessed_input_terms(preprocessing))])
        f.write('## 4. Physical-regime identification\n\n')
        if not regime_detection or regime_detection.get('status') in {None, 'not_requested'}:
            f.write('Mechanism screening was not requested.\n\n')
        else:
            detection_status = regime_detection.get('status', '—')
            selected_k = regime_detection.get('best_k', 1 if detection_status == 'global_analysis_selected' else '—')
            single = regime_detection.get('single_regime_baseline', {}) or {}
            acceptance = regime_detection.get('acceptance_criteria', {}) or {}
            candidates = regime_detection.get('candidate_summary', []) or []
            usable_gradients = len(regime_detection.get('valid_indices', []) or [])
            excluded_gradients = regime_detection.get('excluded_directionless_samples', len(regime_detection.get('excluded_directionless_indices', []) or []))
            write_table(f, [('Decision', detection_status), ('Selected K', selected_k), ('Selection basis', regime_detection.get('selection_basis', '—')), ('Requested K', regime_detection.get('requested_cluster_counts', [])), ('All requested K attempted before selection', (regime_detection.get('decision_evidence') or {}).get('all_requested_K_evaluated_before_selection', '—')), ('Admissible K', (regime_detection.get('decision_evidence') or {}).get('admissible_cluster_counts', '—')), ('K=2 directional support available', (regime_detection.get('decision_evidence') or {}).get('k2_directional_support_available', '—')), ('K=2 actually converged', (regime_detection.get('decision_evidence') or {}).get('k2_actually_converged', '—')), ('K=2 direction action', regime_detection.get('k2_direction_action', '—')), ('Target transform', regime_detection.get('target_transform', '—')), ('Usable gradients', usable_gradients), ('Minimum cluster samples required', acceptance.get('minimum_cluster_samples', '—')), ('Auto multiple support requires K=2 absolute cosine <', acceptance.get('maximum_intercluster_cosine', '—')), ('Excluded gradients', excluded_gradients), ('Low-confidence fraction @ 0.05', regime_detection.get('low_confidence_fraction_at_0.05', '—')), ('Feature coordinate system', regime_detection.get('feature_coordinate_system', '—')), ('Target assumption', regime_detection.get('target_assumption', '—'))])
            if single:
                f.write('### Single-regime baseline\n\n')
                write_table(f, [('Eigenvalues', md_vec(single.get('eigenvalues', []))), ('λ1 / Σλ', single.get('leading_fraction', '—')), ('Global spectrum role', single.get('role', 'diagnostic_only_no_decision_threshold'))])
            failures = regime_detection.get('clustering_failures', []) or []
            if failures:
                f.write('### Requested K without usable partition evidence\n\n')
                write_wide_table(f, ['K', 'Status'], [['K=' + str(failure.get('k', '—')), failure.get('status') or 'No valid candidate partition returned'] for failure in failures])
            if candidates:
                f.write('### Candidate partition evidence\n\n')
                require_convergence = bool(acceptance.get('require_convergence', True))
                rows = []
                for candidate in sorted(candidates, key=lambda item: int(item.get('k', 0))):
                    k = int(candidate.get('k', 0))
                    admissible = bool(candidate.get('meets_regime_criteria', False)) and (bool(candidate.get('converged', False)) or not require_convergence)
                    selected_mark = 'yes' if regime_detection.get('best_k') == k and detection_status == 'success' else ''
                    rows.append([f'K={k}', selected_mark, candidate.get('converged', '—'), md_pct(candidate.get('minimum_cluster_fraction')), candidate.get('passes_fraction_criterion', '—'), candidate.get('minimum_cluster_samples', '—'), candidate.get('passes_sample_count_criterion', '—'), candidate.get('minimum_eigenvalue_ratio', '—'), candidate.get('passes_eigenvalue_ratio_criterion', '—'), candidate.get('max_intercluster_abs_cosine', '—'), candidate.get('minimum_intercluster_angle_degrees', '—'), candidate.get('k2_direction_action', 'not recorded'), 'admissible' if admissible else 'reject', candidate.get('clustering_objective', '—'), candidate.get('termination_reason', '—'), candidate.get('iterations', '—')])
                write_wide_table(f, headers=['K', 'Selected', 'Converged', 'Min fraction', 'Fraction gate', 'Min samples', 'Sample-count gate', 'Min λmax/λmin', 'Ratio gate', 'K=2 |cos θ|', 'K=2 angle (deg)', 'K=2 action', 'Base admissibility', 'Objective', 'Termination', 'Iterations'], rows=rows)
                for candidate in sorted(candidates, key=lambda item: int(item.get('k', 0))):
                    k = int(candidate.get('k', 0))
                    selected_suffix = ' (selected)' if regime_detection.get('best_k') == k and detection_status == 'success' else ''
                    f.write(f'#### K={k}{selected_suffix}\n\n')
                    write_table(f, [('Converged', candidate.get('converged', '—')), ('Minimum cluster fraction', md_pct(candidate.get('minimum_cluster_fraction'))), ('Minimum cluster samples', candidate.get('minimum_cluster_samples', '—')), ('Passes sample-count criterion', candidate.get('passes_sample_count_criterion', '—')), ('K=2 absolute cosine', candidate.get('max_intercluster_abs_cosine', '—')), ('K=2 angle (deg)', candidate.get('minimum_intercluster_angle_degrees', '—')), ('K=2 direction action', candidate.get('k2_direction_action', 'not recorded')), ('Minimum eigenvalue ratio', candidate.get('minimum_eigenvalue_ratio', '—')), ('Meets regime criteria', candidate.get('meets_regime_criteria', '—')), ('Objective', candidate.get('clustering_objective', '—')), ('Termination', candidate.get('termination_reason', '—')), ('Iterations', candidate.get('iterations', '—'))])
                    direction_pairs = (candidate.get('direction_separation') or {}).get('pairs', [])
                    if direction_pairs:
                        write_wide_table(f, ['Cluster pair', 'Absolute cosine', 'Angle (deg)'], [[f"R{pair['cluster_i'] + 1}–R{pair['cluster_j'] + 1}", pair['absolute_cosine'], pair['angle_degrees']] for pair in direction_pairs])
                    clusters = candidate.get('clusters', []) or []
                    if clusters:
                        cluster_rows = []
                        for cluster in clusters:
                            regime_id = int(cluster.get('regime_id', cluster.get('cluster', 0))) + 1
                            eigvecs = cluster.get('eigenvectors_in_log_base_pi', []) or []
                            phys_dirs = cluster.get('dimensionless_exponent_directions_in_physical_variables', []) or []
                            leading_eigvec = np.asarray(eigvecs, dtype=float)[:, 0].tolist() if len(eigvecs) > 0 else None
                            leading_phys = phys_dirs[0] if len(phys_dirs) > 0 else None
                            cluster_rows.append([f'R{regime_id}', cluster.get('n_samples', '—'), md_pct(cluster.get('fraction')), cluster.get('eigenvalue_ratio_max_to_min', '—'), md_vec(cluster.get('eigenvalues')), md_vec(leading_eigvec), md_vec(leading_phys)])
                        signed_pi = bool(regime_detection.get('signed_pi_indices'))
                        feature_basis = 'mixed log-Π and raw Π coordinates' if signed_pi else 'log-Π basis'
                        direction_label = 'Mapped basis coefficients' if signed_pi else 'Leading direction (physical vars)'
                        write_wide_table(f, headers=['Regime', 'Samples', 'Fraction', 'λmax / λmin', 'Eigenvalues', f'Leading eigenvector ({feature_basis})', direction_label], rows=cluster_rows)
                        for cluster in clusters:
                            regime_id = int(cluster.get('regime_id', cluster.get('cluster', 0))) + 1
                            f.write(f'**R{regime_id} full eigenvectors in {feature_basis}**\n\n')
                            f.write('```json\n')
                            f.write(json.dumps(cluster.get('eigenvectors_in_log_base_pi', []), ensure_ascii=False, indent=2))
                            f.write('\n```\n\n')
                            direction_basis = 'input exponent basis' if signed_pi else 'physical variables'
                            f.write(f'**R{regime_id} full dimensionless directions in {direction_basis}**\n\n')
                            f.write('```json\n')
                            f.write(json.dumps(cluster.get('dimensionless_exponent_directions_in_physical_variables', []), ensure_ascii=False, indent=2))
                            f.write('\n```\n\n')
        f.write('## 5. Selected results\n\n')
        if not local_workflows:
            f.write('No successful local workflow was recorded.\n\n')
        for local in local_workflows:
            scope = str(local.get('scope_id', 'global')).replace('_', ' ').title()
            final = local.get('final_solution', {}) or {}
            gpr = final.get('gpr_validation', {}) or {}
            f.write(f'### {scope}\n\n')
            similarity_screen = local.get('similarity_screening') or {}
            if similarity_screen.get('status') == 'success':
                f.write('**Similarity-class screening**\n\n')
                write_table(f, [('Requested class', local.get('similarity_type_requested', 'auto')), ('Selected class', similarity_screen.get('selected_similarity_type')), ('Screening k', similarity_screen.get('screening_k')), ('MI, fixed output', similarity_screen.get('fixed_output_mutual_information')), ('MI, joint output', similarity_screen.get('joint_output_mutual_information')), ('Relative MI loss without output scaling', similarity_screen.get('relative_mi_loss_without_output_scaling')), ('Maximum allowed relative MI change', similarity_screen.get('max_relative_mi_change')), ('Screening cosine threshold', similarity_screen.get('screening_max_cosine_similarity'))])
            write_table(f, [('Status', local.get('status', '—')), ('Selected count', local.get('locked_count', '—')), ('Counts tested', local.get('tested_counts', [])), ('Similarity', str(local.get('similarity_type', '—')).replace('_', ' ')), ('Mutual information', final.get('mutual_information', '—')), ('Normalized information score ε', final.get('normalized_information_score', '—')), ('Held-out R²', gpr.get('test_r2_mean', final.get('gpr_test_r2', '—'))), ('Sparsification', (local.get('sparsification') or {}).get('status', '—')), ('Rationalization', 'accepted' if (local.get('rationalization') or {}).get('accepted') else (local.get('rationalization') or {}).get('status', 'not adopted'))])
            input_formulas = final.get('input_formulas', []) or []
            basis_formulas = final.get('input_basis_coordinate_formulas_display', []) or final.get('input_basis_expanded_formulas_display', []) or []
            output_formula = final.get('output_dimensionless_formula')
            f.write('**Physical-variable relation(s)**\n\n')
            f.write('```text\n' + ('\n'.join(map(str, input_formulas)) if input_formulas else '—') + '\n```\n\n')
            if basis_formulas:
                f.write('**Buckingham–Pi coordinate form(s)**\n\n')
                f.write('```text\n' + '\n'.join(map(str, basis_formulas)) + '\n```\n\n')
            if output_formula:
                f.write('**Output scaling**\n\n')
                f.write(f'```text\n{output_formula}\n```\n\n')
            uncertainty = _plotting._candidate_uncertainty(final)
            diagnostics = {'lock_reason': local.get('lock_reason'), 'count_decisions': local.get('count_decisions', []), 'restart_statistics': {'selection_mode': final.get('selection_mode'), 'restart_seeds': final.get('restart_seeds', []), 'restart_count': final.get('restart_count'), 'scope': uncertainty['restart_scope'], 'mi_std': uncertainty['restart_mi_std'], 'score_std': uncertainty['restart_score_std']}, 'subsampling_statistics': {'scope': uncertainty['subsampling_scope'], 'mi_std': uncertainty['subsample_mi_std'], 'score_std': uncertainty['subsample_score_std']}, 'combined_uncertainty': {'combination_rule': 'sqrt(restart_std^2 + subsample_std^2)', 'mi_std': uncertainty['combined_mi_std'], 'score_std': uncertainty['combined_score_std']}, 'similarity_screening': local.get('similarity_screening', {}), 'review_flags': local.get('review_flags', {})}
            f.write('<details>\n<summary>Diagnostics</summary>\n\n```json\n')
            f.write(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str))
            f.write('\n```\n\n</details>\n\n')
        final_report = state.get('final_report')
        if final_report:
            f.write('## 6. Model output\n\n')
            f.write(md_text(final_report, 5000) + '\n\n')
            interpretation_rag = state.get('interpretation_knowledge_retrieval', {}) or {}
            if interpretation_rag:
                f.write('### Dimensionless-relation retrieval provenance\n\n')
                write_table(f, [('Retrieval status', interpretation_rag.get('status', '—')), ('Knowledge base', interpretation_rag.get('kb_path', '—')), ('Knowledge-base SHA256', interpretation_rag.get('kb_sha256', '—')), ('Embedding provider', interpretation_rag.get('embedding_provider', '—')), ('Embedding model', interpretation_rag.get('embedding_model', '—')), ('Embedding base URL', interpretation_rag.get('embedding_base_url', '—')), ('Embedding cache', interpretation_rag.get('embedding_cache_path', '—')), ('Embedding status', interpretation_rag.get('embedding_status', '—')), ('Retrieval mode', interpretation_rag.get('retrieval_mode', '—')), ('Retrieved records', interpretation_rag.get('n_retrieved_records', 0)), ('Interpretation top-k', interpretation_rag.get('top_k', globals().get('KNOWLEDGE_INTERPRETATION_TOP_K', '—'))), ('Lexical fallback allowed', interpretation_rag.get('allow_lexical_fallback', globals().get('KNOWLEDGE_ALLOW_LEXICAL_FALLBACK', '—')))])
                interpretation_rows = []
                for rank, match in enumerate(interpretation_rag.get('matches', []) or [], start=1):
                    interpretation_rows.append([rank, match.get('canonical_name', '—'), match.get('symbols', '—'), match.get('aliases', '—'), match.get('defining_expression', '—'), match.get('physical_interpretation', '—'), match.get('domain', '—'), match.get('retrieval_score', '—'), match.get('lexical_score', '—'), match.get('semantic_score', '—'), match.get('retrieval_mode', '—'), match.get('source', '—'), match.get('source_identifier', '—'), match.get('confidence', '—')])
                if interpretation_rows:
                    write_wide_table(f, headers=['Rank', 'Canonical name', 'Symbols', 'Aliases', 'Defining expression', 'Physical interpretation', 'Domain', 'Retrieval score', 'Lexical score', 'Semantic score', 'Retrieval mode', 'Source', 'Source identifier', 'Confidence'], rows=interpretation_rows)
                write_rag_evidence_block(f, 'Dimensionless-relation evidence supplied to Model output', interpretation_rag.get('evidence_markdown', ''))
        figures = state.get('visualization_files', []) or []
        if figures:
            f.write('## 7. Generated figures\n\n')
            for figure in figures:
                f.write(f'- `{figure}`\n')
            f.write('\n')
        try:
            _write_run_provenance_markdown(f, state)
        except Exception as recording_error:
            print('[Recording warning] Markdown provenance: ' + type(recording_error).__name__)
    print(f'✅ State saved to markdown: {save_path}')
    return save_path


_FRONTEND_PRESENTATION = {}


def _frontend_text(value):
    if value is None:
        return 'Not recorded'
    if isinstance(value, (list, tuple)):
        return ', '.join(_frontend_text(x) for x in value) if value else 'None'
    if isinstance(value, dict):
        return '; '.join(str(k) + ': ' + _frontend_text(v) for k, v in value.items()) if value else 'None'
    return str(value)


def _frontend_escape(value):
    return html.escape(_frontend_text(value)).replace('\n', '<br>')


def _frontend_badge(value):
    label = str(value or 'Pending')
    key = label.upper()
    tone = 'pass' if key in {'PASSED', 'COMPLETE', 'COMPLETED'} else 'fail' if key in {'BLOCKED', 'FAILED', 'CONFLICT'} else 'neutral'
    return "<span class='daa-mini-badge " + tone + "'>" + html.escape(label) + '</span>'


def _frontend_table(headers, rows, matrix=False):
    def cell(value):
        if isinstance(value, tuple) and len(value) == 2 and value[0] == 'html':
            return value[1]
        return _frontend_escape(value)
    classes = 'daa-diagnostic-table daa-front-table' + (' daa-front-matrix' if matrix else '')
    return "<div class='daa-front-scroll'><table class='" + classes + "'><thead><tr>" + ''.join('<th>' + html.escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _frontend_details(title, body, opened=False):
    return "<details class='daa-details'" + (' open' if opened else '') + '><summary>' + html.escape(str(title)) + '</summary>' + body + '</details>'


def _frontend_raw(title, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return _frontend_details(title, '<pre>' + html.escape(text) + '</pre>')


def _frontend_si(vector):
    if not isinstance(vector, dict):
        return 'Not provided'
    terms = []
    for base in ('M', 'L', 'T', 'I', 'Theta', 'N', 'J'):
        value = vector.get(base)
        if value is None:
            terms.append(base + '^?')
        elif str(value) not in {'0', '0.0', '-0', '-0.0'}:
            terms.append(base if str(value) == '1' else base + '^(' + str(value) + ')')
    return ' '.join(terms) or '1 (dimensionless)'


def _frontend_si_html(vector):
    if not isinstance(vector, dict):
        return "<span class='daa-front-si'>Not provided</span>"
    terms = []
    for base in ('M', 'L', 'T', 'I', 'Theta', 'N', 'J'):
        symbol = '\u0398' if base == 'Theta' else base
        value = vector.get(base)
        if value is None:
            terms.append(symbol + '<sup>?</sup>')
        elif str(value) not in {'0', '0.0', '-0', '-0.0'}:
            terms.append(symbol if str(value) == '1' else symbol + '<sup>' + html.escape(str(value)) + '</sup>')
    return "<span class='daa-front-si'>" + ' '.join(terms or ['1 (dimensionless)']) + '</span>'


def _frontend_ledger_html(ledger, contract=None):
    if not isinstance(ledger, dict):
        return "<div class='daa-note'>The Analyzer has not returned a usable physical-input ledger.</div>"
    variables = ledger.get('variables') or []
    by_name = {v.get('csv_name'): v for v in variables if isinstance(v, dict)}
    names = list(ledger.get('input_variable_names') or [])
    rows, exact_rows = [], []
    for name in names:
        v = by_name.get(name, {})
        meaning = '<strong>' + _frontend_escape(v.get('quantity_name')) + '</strong><div class="daa-front-reason">' + _frontend_escape(v.get('physical_definition')) + '</div>'
        rows.append([name, ('html', meaning), ('html', _frontend_si_html(v.get('si_exponents')))])
        powers = v.get('si_exponents') or {}
        exact_rows.append([name] + [('?' if powers.get(b) is None else powers[b]) for b in ('M', 'L', 'T', 'I', 'Theta', 'N', 'J')])
    result = "<div class='daa-section-title'>Physical input ledger</div>" + _frontend_table(['CSV input', 'Physical meaning', 'Named SI dimensions'], rows)
    result += "<div class='daa-front-caption'>? means an unknown coordinate; it is never replaced by zero. A value of 0 is known absence of that SI base.</div>"
    result += _frontend_details('All seven named SI exponents', _frontend_table(['Input', 'M', 'L', 'T', 'I', 'Theta', 'N', 'J'], exact_rows, matrix=True))
    output = ledger.get('output_variable_name')
    result += "<div class='daa-note'><strong>Output " + _frontend_escape(output) + ':</strong> ' + _frontend_escape(ledger.get('output_variable_reasoning')) + '</div>'
    if isinstance(contract, dict):
        D, axes, columns = contract.get('D_exact'), contract.get('row_order') or [], contract.get('var_names') or []
        result += "<div class='daa-section-title'>Program-compiled dimensional matrix</div>"
        if D is not None and len(D) == len(axes) and all(len(row) == len(columns) for row in D):
            result += _frontend_table(['SI base / input'] + columns, [[axis] + list(row) for axis, row in zip(axes, D)], matrix=True)
            result += "<div class='daa-front-caption'>Rows and columns follow the labels above; zero rows may be omitted. Counts are computed from this matrix, not inferred separately by the model.</div>"
        else:
            result += "<div class='daa-note'>No numeric matrix was compiled: named coordinates remain unresolved or the matrix is not available.</div>"
    return result


def _frontend_query_rows(query):
    prefixes = ('Target CSV variable:', 'Local problem description:', 'Problem domain/context:', 'Analyzer interpretation:')
    fields = []
    for line in str(query).splitlines():
        prefix = next((p for p in prefixes if line.startswith(p)), None)
        if prefix:
            fields.append([prefix[:-1], line[len(prefix):].lstrip()])
        elif fields:
            fields[-1][1] += '\n' + line
        else:
            fields.append(['Query text', line])
    return fields


def _frontend_record_html(record, visible=True):
    title = record.get('canonical_quantity') or 'Retrieved physical quantity'
    identifier = record.get('entry_id', 'Not recorded')
    rows = [(label, record.get(key)) for label, key in (
        ('Entry ID', 'entry_id'), ('Definition', 'definition'), ('SI dimensions', 'si_dimension'),
        ('Vector [M,L,T,I,Theta,N,J]', 'dimension_vector'), ('SI unit', 'si_unit'),
        ('Domain', 'domain'), ('Ambiguity notes', 'ambiguity_notes'), ('Source', 'source'),
        ('Source identifier', 'source_identifier'), ('Lexical score', 'lexical_score'),
        ('Semantic score', 'semantic_score'), ('Hybrid score', 'hybrid_score'), ('Retrieval mode', 'retrieval_mode'))]
    body = '<article class="daa-result"><h4>' + _frontend_escape(title) + '</h4>'
    body += "<div class='daa-front-caption'>" + _frontend_escape(identifier) + (' · visible to Checker' if visible else ' · not included in the Checker-visible evidence limit') + '</div>'
    body += _frontend_table(['Record field', 'Retrieved value'], rows) + '</article>'
    return body


def _frontend_retrieval_html(retrieval):
    if not isinstance(retrieval, dict) or not retrieval:
        return "<div class='daa-front-caption'>Retrieval has not completed for this audit round.</div>"
    metadata = [('Status', retrieval.get('status')), ('Modes', retrieval.get('retrieval_modes')),
                ('Embedding status', retrieval.get('embedding_status')), ('Retrieved records', retrieval.get('n_retrieved_records', 0))]
    result = _frontend_table(['Retrieval', 'Observed value'], metadata)
    if retrieval.get('error'):
        result += "<div class='daa-note'>" + _frontend_escape(retrieval['error']) + '</div>'
    queries = retrieval.get('queries_by_variable') or {}
    matches = retrieval.get('matches_by_variable') or {}
    visible = retrieval.get('evidence_records_visible_by_variable')
    sections = []
    for name in dict.fromkeys(list(queries) + list(matches)):
        part = "<div class='daa-section-title'>Input " + _frontend_escape(name) + '</div>'
        if name in queries:
            part += _frontend_table(['Actual query field', 'Exact query value'], _frontend_query_rows(queries[name]))
            part += "<div class='daa-front-caption'>Query SHA256: " + _frontend_escape((retrieval.get('query_sha256_by_variable') or {}).get(name)) + ' · Observed in retriever: ' + _frontend_escape((retrieval.get('query_observed_in_retriever_by_variable') or {}).get(name)) + '</div>'
        cards = []
        for record in matches.get(name, []) or []:
            is_visible = visible is None or str(record.get('entry_id', '')) in visible.get(name, [])
            cards.append(_frontend_record_html(record, is_visible))
        part += "<div class='daa-result-grid'>" + ''.join(cards) + '</div>' if cards else "<div class='daa-front-caption'>No sufficiently relevant record was retrieved; absence alone is not a physical contradiction.</div>"
        sections.append(part)
    if sections:
        result += _frontend_details('Actual retrieval queries and reference records', ''.join(sections))
    evidence = retrieval.get('evidence_markdown')
    if evidence:
        result += _frontend_details('Exact evidence supplied to Checker', '<div class="daa-formula">' + _frontend_escape(evidence) + '</div>')
    return result


def _frontend_audit_html(audit, raw=None):
    audit = audit if isinstance(audit, dict) else {}
    checker = audit.get('checker_audit') or raw or {}
    checker = checker if isinstance(checker, dict) else {}
    originals = {v.get('csv_name'): v for v in checker.get('variables', []) if isinstance(v, dict)}
    results = audit.get('variable_results') or []
    if not results:
        results = [{'csv_name':v.get('csv_name'), 'decision':v.get('decision'), 'before':v.get('before_si_exponents'), 'suggested':v.get('suggested_si_exponents'), 'referenced_entry_ids':v.get('evidence_entry_ids', [])} for v in checker.get('variables', []) if isinstance(v, dict)]
    rows, reasons, program_notes = [], [], []
    for v in results:
        name = v.get('csv_name'); original = originals.get(name, {})
        before = v.get('before', original.get('before_si_exponents'))
        suggested = original.get('suggested_si_exponents')
        effective = v.get('suggested')
        proposed = _frontend_si_html(suggested)
        if effective is not None and effective != suggested and effective != before:
            proposed += '<div class="daa-front-caption">Program correction: ' + _frontend_si_html(effective) + '</div>'
        differences = v.get('coordinate_diff')
        if differences is None:
            differences = {b:{'before':before.get(b), 'suggested':suggested.get(b)} for b in ('M','L','T','I','Theta','N','J') if isinstance(before,dict) and isinstance(suggested,dict) and before.get(b)!=suggested.get(b)}
        diff_text = '; '.join(str(b) + ': ' + str(d.get('before')) + ' → ' + str(d.get('suggested')) for b, d in differences.items()) if differences else 'No coordinate change'
        support = original.get('support_mode', 'Not recorded')
        ids = v.get('referenced_entry_ids', original.get('evidence_entry_ids', []))
        rows.append([name, ('html', _frontend_badge(v.get('decision'))), ('html', _frontend_si_html(before)), ('html', proposed), diff_text, support + (' · ' + ', '.join(map(str, ids)) if ids else '')])
        reason = original.get('reason')
        issues = v.get('issues') or []
        conflicts = v.get('known_coordinate_conflicts_with_reference') or {}
        if issues or conflicts:
            text = _frontend_escape(issues)
            if conflicts:
                text += '<br><strong>Known conflict with the cited KB vector:</strong> ' + '; '.join(html.escape(str(base)) + ': candidate/suggestion ' + html.escape(str(values.get('candidate_or_suggestion'))) + ', cited record ' + html.escape(str(values.get('cited_record'))) for base, values in conflicts.items())
            program_notes.append('<div class="daa-note"><strong>Program checks for ' + _frontend_escape(name) + ':</strong> ' + text + '</div>')
        reasons.append('<article class="daa-result"><h4>' + _frontend_escape(name) + '</h4>' + ('<p>' + _frontend_escape(reason) + '</p>' if reason else '<p>No structured Checker reason was returned.</p>') + ('<p><strong>Program issues:</strong> ' + _frontend_escape(issues) + '</p>' if issues else '') + '</article>')
    result = _frontend_table(['Input', 'Checker decision', 'Candidate SI exponents', 'Suggested SI', 'Coordinate changes', 'Evidence / support'], rows) if rows else "<div class='daa-note'>No usable structured variable audit is available. The program decision and failure reason remain recorded below.</div>"
    result += "<div class='daa-front-caption'>Checker decisions are model judgments. The program gate is reported separately above; ? denotes an unknown coordinate, never zero.</div>" + ''.join(program_notes)
    if reasons:
        result += _frontend_details('Per-variable physical reasoning and program checks', "<div class='daa-result-grid'>" + ''.join(reasons) + '</div>')
    feedback = audit.get('feedback') or '\n'.join(map(str, audit.get('issues', [])))
    if feedback:
        result += _frontend_details('Exact feedback returned to Analyzer', '<div class="daa-formula">' + _frontend_escape(feedback) + '</div>', opened=audit.get('gate_status') in {'RETRY','BLOCKED'})
    return result


def _frontend_context(stage):
    if stage not in _FRONTEND_PRESENTATION:
        _FRONTEND_PRESENTATION[stage] = {'stage':stage, 'round':None, 'status':'running', 'handle':None, 'extras':[]}
    return _FRONTEND_PRESENTATION[stage]


def _frontend_render(stage, context):
    title = 'Problem formulation' if stage == '01' else 'Physical consistency'
    status = context.get('status', 'running')
    tone, label, status_class = _status_style(status)
    rows = [('Analysis round' if stage == '01' else 'Audit round', context.get('round'))]
    body = ''
    if stage == '01':
        compiled = context.get('compiled') or {}
        ledger = compiled.get('ledger') or context.get('raw_analyzer')
        contract = compiled.get('contract')
        if isinstance(ledger,dict):
            rows += [('Output', ledger.get('output_variable_name')), ('Input order', ledger.get('input_variable_names'))]
            if contract:
                rows += [('n / rank r / k', str(contract.get('n')) + ' / ' + str(contract.get('r')) + ' / ' + str(contract.get('k'))), ('Ledger status', contract.get('status'))]
            body += _frontend_ledger_html(ledger, contract)
        else:
            body += "<div class='daa-summary'>Identifying the output, physical input meanings and named SI dimensions.</div>"
        if compiled:
            body += _frontend_details('Ledger identity and advanced structured diagnostics', _frontend_table(['Field','Value'], [('Ledger SHA256',compiled.get('ledger_sha256'))]) + _frontend_raw('Raw Analyzer and compiled contract', {'analyzer':context.get('raw_analyzer'),'compiled':compiled}))
    else:
        retrieval = context.get('retrieval') or {}
        audit = context.get('audit') or {}
        rows += [('Program decision', audit.get('gate_status','Pending')), ('Retrieved records',retrieval.get('n_retrieved_records',0)), ('Retrieval status',retrieval.get('status','Pending'))]
        if audit:
            rows.append(('Revision limit exhausted', audit.get('retry_budget_exhausted',False)))
        body += "<div class='daa-section-title'>Variable-by-variable physical audit</div>" + _frontend_audit_html(audit, context.get('raw_checker'))
        body += "<div class='daa-section-title'>Retrieval-grounded evidence</div>" + _frontend_retrieval_html(retrieval)
        body += _frontend_details('Advanced structured diagnostics', _frontend_table(['Field','Value'],[('Ledger SHA256',audit.get('ledger_sha256'))]) + _frontend_raw('Raw Checker response and program gate',{'checker':context.get('raw_checker'),'gate':audit}))
    if context.get('failure'):
        body += '<div class="daa-note"><strong>Failure:</strong> ' + _frontend_escape(context['failure']) + '</div>'
    metrics = "<div class='daa-metrics'>" + ''.join('<div class="daa-metric"><div class="daa-metric-key">' + _frontend_escape(k) + '</div><div class="daa-metric-value">' + _frontend_escape(v) + '</div></div>' for k,v in rows) + '</div>'
    return "<div class='daa-shell daa-frontend' data-frontend-stage='" + stage + "'><section class='daa-card' data-tone='" + tone + "'><div class='daa-card-head'><div><div class='daa-card-label'>Stage " + stage + "</div><h3 class='daa-card-title'>" + title + "</h3></div><span class='daa-status " + status_class + "'>" + html.escape(label) + '</span></div>' + metrics + body + '</section></div>'


def _frontend_update(stage):
    context = _frontend_context(stage)
    if context.get('pending_header') and context.get('status') == 'running':
        return
    context['pending_header'] = False
    _inject_agent_editorial_styles()
    card = HTML(_frontend_render(stage, context))
    handle = context.get('handle')
    if handle is not None and hasattr(handle,'update'):
        handle.update(card)
    else:
        context['handle'] = display(card, display_id=True)


def _frontend_flush_pending():
    """Emit the first Stage01 slot only after the original run header is shown."""
    try:
        context = _FRONTEND_PRESENTATION.get('01')
        if context and context.get('pending_header'):
            context['pending_header'] = False
            _frontend_update('01')
    except Exception as exc:
        print('[Display warning] Pending front-end card: ' + str(exc))


def display_frontend_stage(title, *, hyperparameters=None, **options):
    """Render only; a display failure never changes scientific execution."""
    try:
        if not _workflow_ui_enabled(hyperparameters):
            return
        stage = '01' if ('1/7' in title or 'Stage 01' in title) else '02'
        status = options.get('status','running')
        if status == 'running':
            round_value = next((v for k,v in options.get('rows',[]) if 'round' in str(k).lower() or 'attempt' in str(k).lower()),None)
            _FRONTEND_PRESENTATION[stage] = {'stage':stage,'round':round_value,'status':status,'handle':None,'extras':[], 'pending_header':stage == '01' and round_value == 1}
        context = _frontend_context(stage)
        context['status'] = status
        if status == 'failed':
            context['failure'] = options.get('summary','The node did not complete.')
        _frontend_update(stage)
    except Exception as exc:
        print('[Display warning] ' + str(title) + ': ' + str(exc))


def display_frontend_details(title, value, hyperparameters=None):
    """Capture the actual payload for formatted display; raw JSON stays closed."""
    try:
        if not _workflow_ui_enabled(hyperparameters):
            return
        if title == 'Analyzer structured response':
            _frontend_context('01')['raw_analyzer'] = value
            _frontend_update('01')
        elif title == 'Checker structured response':
            _frontend_context('02')['raw_checker'] = value
            _frontend_update('02')
        else:
            display(HTML("<div class='daa-shell daa-frontend'>" + _frontend_raw(title,value) + '</div>'))
    except Exception as exc:
        print('[Display warning] ' + str(title) + ': ' + str(exc))


def display_analysis_result(compiled, hyperparameters=None):
    try:
        if not _workflow_ui_enabled(hyperparameters):
            return
        context = _frontend_context('01')
        context['compiled'] = compiled
        context['status'] = 'completed' if compiled['contract']['status'] == 'complete' else 'warning'
        _frontend_update('01')
    except Exception as exc:
        print('[Display warning] Analyzer result: ' + str(exc))


def display_retrieval_result(retrieval, hyperparameters=None):
    try:
        if not _workflow_ui_enabled(hyperparameters):
            return
        context = _frontend_context('02')
        context['retrieval'] = retrieval
        _frontend_update('02')
    except Exception as exc:
        print('[Display warning] Retrieval result: ' + str(exc))


def display_audit_result(audit, hyperparameters=None):
    try:
        if not _workflow_ui_enabled(hyperparameters):
            return
        context = _frontend_context('02')
        context['audit'] = audit
        decision = audit.get('gate_status','BLOCKED')
        context['status'] = 'passed' if decision == 'PASSED' else 'warning' if decision == 'RETRY' else 'failed'
        _frontend_update('02')
    except Exception as exc:
        print('[Display warning] Audit result: ' + str(exc))


def _write_frontend_markdown(handle, state):
    """Readable report tables; source records and feedback remain complete."""
    def cell(value):
        return _frontend_text(value).replace('|', '\\|').replace('\n','<br>')
    def table(headers, rows):
        handle.write('| ' + ' | '.join(map(cell,headers)) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n')
        for row in rows:
            handle.write('| ' + ' | '.join(map(cell,row)) + ' |\n')
        handle.write('\n')
    ledger = state.get('dimension_ledger') or {}
    contract = state.get('dimension_contract') or {}
    retrieval = state.get('knowledge_retrieval') or {}
    handle.write('## 2. Dimensional formulation and physical audit\n\n')
    table(['Field','Value'],[('Output',ledger.get('output_variable_name',state.get('analyzed_output_var'))),('Output reasoning',ledger.get('output_variable_reasoning',state.get('analyzed_output_reasoning'))),('Input order',ledger.get('input_variable_names',state.get('analyzed_input_vars'))),('n / r / k',str(contract.get('n'))+' / '+str(contract.get('r'))+' / '+str(contract.get('k'))),('Ledger SHA256',state.get('dimension_ledger_sha256'))])
    if ledger.get('variables'):
        handle.write('### Physical input ledger\n\n')
        table(['CSV input','Physical quantity','Physical definition','Named SI dimensions'],[[v.get('csv_name'),v.get('quantity_name'),v.get('physical_definition'),_frontend_si(v.get('si_exponents'))] for v in ledger['variables']])
        handle.write('? means unknown, never zero.\n\n')
    if contract.get('D_exact') is not None:
        handle.write('### Compiled dimensional matrix\n\n')
        table(['SI base / input']+list(contract.get('var_names') or []),[[b]+list(row) for b,row in zip(contract.get('row_order') or [],contract['D_exact'])])
    else:
        handle.write('No numeric dimensional matrix is available for unresolved coordinates.\n\n')
    history = state.get('dimension_audit_history') or ([state['dimension_audit']] if state.get('dimension_audit') else [])
    for number,audit in enumerate(history,1):
        handle.write('### Physical audit round '+str(number)+'\n\n')
        table(['Field','Value'],[('Program decision',audit.get('gate_status')),('Revision limit exhausted',audit.get('retry_budget_exhausted',False)),('Ledger SHA256',audit.get('ledger_sha256'))])
        checker = {v.get('csv_name'):v for v in (audit.get('checker_audit') or {}).get('variables',[])}
        table(['Input','Checker decision','Candidate SI exponents','Suggested SI','Coordinate changes','Evidence IDs / reason'],[[v.get('csv_name'),v.get('decision'),_frontend_si(v.get('before')),_frontend_si(v.get('suggested')),'; '.join(str(b)+': '+str(d.get('before'))+' -> '+str(d.get('suggested')) for b,d in (v.get('coordinate_diff') or {}).items()),_frontend_text(v.get('referenced_entry_ids'))+'; '+str(checker.get(v.get('csv_name'),{}).get('reason','Not recorded'))] for v in audit.get('variable_results',[])])
        feedback=audit.get('feedback') or '\n'.join(map(str,audit.get('issues',[])))
        if feedback:
            handle.write('<details><summary>Exact feedback returned to Analyzer</summary>\n\n'+str(feedback)+'\n\n</details>\n\n')
    handle.write('### Actual retrieval evidence\n\n')
    table(['Field','Observed value'],[('Status',retrieval.get('status')),('Modes',retrieval.get('retrieval_modes')),('Embedding status',retrieval.get('embedding_status')),('Retrieved records',retrieval.get('n_retrieved_records'))])
    for name,query in (retrieval.get('queries_by_variable') or {}).items():
        handle.write('#### Input '+str(name)+' — exact query\n\n')
        table(['Query field','Actual value'],_frontend_query_rows(query))
        table(['Query identity','Value'],[('SHA256',(retrieval.get('query_sha256_by_variable') or {}).get(name)),('Observed in retriever',(retrieval.get('query_observed_in_retriever_by_variable') or {}).get(name))])
    for name,records in (retrieval.get('matches_by_variable') or {}).items():
        for record in records:
            handle.write('#### '+str(name)+' — '+str(record.get('canonical_quantity','Retrieved record'))+'\n\n')
            table(['Record field','Retrieved value'],[(k,record.get(k)) for k in ('entry_id','definition','si_dimension','dimension_vector','si_unit','domain','ambiguity_notes','source','source_identifier','lexical_score','semantic_score','hybrid_score','retrieval_mode')])
    evidence=retrieval.get('evidence_markdown')
    if evidence:
        handle.write('<details><summary>Exact Checker-visible evidence</summary>\n\n'+str(evidence)+'\n\n</details>\n\n')
    handle.write('<details><summary>Advanced structured front-end diagnostics</summary>\n\n```json\n'+json.dumps({'ledger':ledger,'contract':contract,'audit_history':history},ensure_ascii=False,indent=2,default=str)+'\n```\n\n</details>\n\n')


def display_dimagent_graph(graph=None):
    """Display the graph with LangGraph's native Mermaid PNG renderer."""
    graph = __import__(__package__ + ".graph", fromlist=["build_agent_graph"]).build_agent_graph() if graph is None else graph
    display(Image(graph.get_graph().draw_mermaid_png()))


def _recorded_costs_summary(provenance):
    """Aggregate recorded fields only; unknown usage never becomes a free call."""
    attempts = list(provenance.get("http_attempts") or [])

    def summarize(rows):
        known = [row.get("total_tokens") for row in rows]
        return {
            "attempts": len(rows),
            "successes": sum(row.get("outcome") == "http_success" for row in rows),
            "failures": sum(row.get("outcome") in {"http_error", "transport_error"} for row in rows),
            "input_tokens": sum(row["input_tokens"] for row in rows if row.get("input_tokens") is not None),
            "output_tokens": sum(row["output_tokens"] for row in rows if row.get("output_tokens") is not None),
            "total_tokens": sum(value for value in known if value is not None),
            "missing_input": sum(row.get("input_tokens") is None for row in rows),
            "missing_output": sum(row.get("output_tokens") is None for row in rows),
            "missing_total": sum(value is None for value in known),
            "seconds": sum(float(row.get("duration_seconds") or 0.0) for row in rows),
            "missing_seconds": sum(row.get("duration_seconds") is None for row in rows),
            "known_retry_attempts": sum((row.get("sdk_retry_index") or 0) > 0 for row in rows),
        }

    groups = {}
    for row in attempts:
        key = (row.get("node") or "Unattributed", row.get("kind") or "unknown", row.get("requested_model") or "Not recorded")
        groups.setdefault(key, []).append(row)
    return {
        "all": summarize(attempts),
        "llm": summarize([row for row in attempts if row.get("kind") == "llm"]),
        "embedding": summarize([row for row in attempts if row.get("kind") == "embedding"]),
        "groups": [(key, summarize(rows)) for key, rows in groups.items()],
    }


def _recorded_token_text(value, missing=0, attempts=0):
    if attempts and missing == attempts:
        return "Not reported"
    text = f"{int(value):,}"
    return text + " + unreported" if missing else text


def _recorded_seconds_text(value, missing=0, attempts=0):
    if value is None or (attempts and missing == attempts):
        return "Not recorded"
    text = f"{float(value):.3f}"
    return text + " + unrecorded" if missing else text


def _recorded_costs_rows(provenance):
    role_names = {
        "Analyzer": "Analyzer", "Hallu_Checker": "Physics audit",
        "Output_Binding_Checker": "Output column check",
        "Workflow_Planner": "Planner", "Workflow_Report": "Model output",
    }
    rows = []
    for (node, kind, model), stats in _recorded_costs_summary(provenance)["groups"]:
        role = role_names.get(node, node)
        if kind == "embedding":
            role += " / RAG embeddings"
        rows.append([
            role, model, stats["attempts"],
            str(stats["successes"]) + " / " + str(stats["failures"]),
            _recorded_token_text(stats["input_tokens"], stats["missing_input"], stats["attempts"]),
            "Not applicable" if kind == "embedding" else _recorded_token_text(stats["output_tokens"], stats["missing_output"], stats["attempts"]),
            _recorded_token_text(stats["total_tokens"], stats["missing_total"], stats["attempts"]),
            _recorded_seconds_text(stats["seconds"], stats["missing_seconds"], stats["attempts"]),
        ])
    return rows


def _recorded_provenance_identity(provenance):
    dataset = provenance.get("dataset") or {}
    rows = [
        ("Run ID", provenance.get("run_id")),
        ("Code release", provenance.get("release")),
        ("Workflow status", provenance.get("workflow_status")),
        ("Numerical result status", provenance.get("scientific_status")),
        ("Started (UTC)", provenance.get("started_utc")),
        ("Finished (UTC)", provenance.get("finished_utc")),
        ("Data file", dataset.get("name")),
        ("Data SHA256", dataset.get("sha256")),
        ("Problem SHA256", provenance.get("problem_sha256")),
        ("Manual-parameter SHA256", provenance.get("manual_hyperparams_sha256")),
        ("Scientific-engine fingerprint", provenance.get("engine_sha256")),
        ("LangSmith tracing configured", provenance.get("tracing_enabled")),
    ]
    for role, model in (provenance.get("model_configuration") or {}).items():
        rows.append((str(role).replace("_", " ").capitalize() + " model", model))
    for package, version in (provenance.get("package_versions") or {}).items():
        rows.append(("Package: " + str(package), version))
    return rows


def _recorded_attempt_rows(provenance):
    return [[
        row.get("attempt_id"), row.get("node"), row.get("kind"),
        row.get("requested_model"), row.get("response_model"),
        row.get("http_status"), row.get("outcome"), row.get("sdk_retry_index"),
        row.get("input_tokens"), row.get("output_tokens"), row.get("total_tokens"),
        row.get("cached_input_tokens"), row.get("reasoning_tokens"),
        _recorded_seconds_text(row.get("duration_seconds")), row.get("error_type"),
        row.get("response_id"), row.get("request_sha256"),
    ] for row in provenance.get("http_attempts") or []]


def _recorded_costs_html(provenance):
    """Use the existing publication card style; keep detailed receipts closed."""
    summary = _recorded_costs_summary(provenance)
    llm, embedding, total = summary["llm"], summary["embedding"], summary["all"]
    status = str(provenance.get("status") or "not_recorded")
    completed = status in {"completed", "success"}
    failed = status in {"failed", "blocked", "terminal_unresolved", "basis_failed"}
    tone, _, badge_class = _status_style("completed" if completed else "failed" if failed else "running")
    http_time = _recorded_seconds_text(total["seconds"], total["missing_seconds"], total["attempts"])
    metrics = [
        ("LLM HTTP attempts", llm["attempts"]),
        ("LLM reported tokens", _recorded_token_text(llm["total_tokens"], llm["missing_total"], llm["attempts"])),
        ("Embedding HTTP attempts", embedding["attempts"]),
        ("Embedding reported tokens", _recorded_token_text(embedding["total_tokens"], embedding["missing_total"], embedding["attempts"])),
        ("Cumulative HTTP time", http_time if http_time == "Not recorded" else http_time + " s"),
        ("Workflow wall time", "Not recorded" if provenance.get("wall_seconds") is None else f"{float(provenance['wall_seconds']):.3f} s"),
    ]
    metrics_html = "<div class='daa-metrics'>" + "".join(
        "<div class='daa-metric'><div class='daa-metric-key'>" + _frontend_escape(key)
        + "</div><div class='daa-metric-value'>" + _frontend_escape(value) + "</div></div>"
        for key, value in metrics
    ) + "</div>"
    rows = _recorded_costs_rows(provenance)
    usage_table = _frontend_table(
        ["Role / service", "Requested model", "HTTP attempts", "2xx / failures", "Input tokens", "Output tokens", "Reported total", "HTTP time (s)"], rows,
    ) if rows else "<div class='daa-note'>No model or embedding HTTP attempts were recorded for this run.</div>"
    missing_note = (
        f"{total['missing_total']} attempt(s) lack a reported total. Known tokens are shown as a partial total; missing usage is not treated as zero cost. "
        if total["missing_total"] else ""
    )
    note = (
        missing_note
        + "HTTP attempts include native SDK retries. A 2xx response records transport success; parsing and physics validation may still fail. "
        + "Input tokens may include cached input; output tokens may include reasoning. These subsets are not added again. "
        + "Cumulative HTTP time excludes retry backoff and can overlap across concurrent requests. Workflow wall time includes numerical execution. "
        + "Only observed model/embedding requests are counted; Mermaid rendering and LangSmith telemetry are outside this ledger. "
        + "Monetary charges are not estimated without provider billing or a configured price schedule."
    )
    timings = _frontend_table(
        ["Workflow node", "Visit", "Status", "Wall time (s)", "Error type"],
        [[item.get("node"), item.get("visit"), item.get("status"), _recorded_seconds_text(item.get("duration_seconds")), item.get("error_type")] for item in provenance.get("node_timings") or []],
    )
    attempt_table = _frontend_table(
        ["Attempt", "Node", "Service", "Requested model", "Returned model", "HTTP", "Outcome", "SDK retry index", "Input", "Output", "Total", "Cached input subset", "Reasoning output subset", "Seconds", "Error type", "Response ID", "Request SHA256"],
        _recorded_attempt_rows(provenance),
    )
    details = _frontend_details(
        "Run identity, configured models and software versions",
        _frontend_table(["Recorded field", "Value"], _recorded_provenance_identity(provenance)),
    ) + _frontend_details("Workflow timing by node", timings) + _frontend_details(
        "Individual HTTP attempts and token breakdown", attempt_table,
    ) + _frontend_raw("Complete provenance record", provenance)
    warnings_html = ""
    if provenance.get("observation_warnings"):
        warnings_html = "<div class='daa-note'><strong>Recording limitations:</strong> " + _frontend_escape(provenance["observation_warnings"]) + "</div>"
    return (
        "<div class='daa-shell daa-recorded-costs'><section class='daa-card' data-tone='" + tone + "'>"
        + "<div class='daa-card-head'><div><div class='daa-card-label'>Run record</div>"
        + "<h3 class='daa-card-title'>Run provenance and recorded costs</h3></div>"
        + "<span class='daa-status " + badge_class + "'>" + _frontend_escape(status.replace("_", " ")) + "</span></div>"
        + "<div class='daa-summary'>Recorded API use, workflow timing and identities for this invocation.</div>"
        + metrics_html + "<div class='daa-section-title'>Model and embedding usage</div>" + usage_table
        + "<div class='daa-note'>" + _frontend_escape(note) + "</div>" + warnings_html + details
        + "</section></div>"
    )


def display_run_provenance(state=None):
    """Display a saved run or the latest partial record, without calling APIs."""
    provenance = _provenance.get_run_provenance(state)
    if not provenance:
        display(HTML("<div class='daa-shell'><div class='daa-note'>No run provenance is available. Invoke the graph to record a new run.</div></div>"))
        return None
    _inject_agent_editorial_styles()
    display(HTML(_recorded_costs_html(provenance)))
    return provenance


def _write_run_provenance_markdown(handle, state):
    """Append readable costs and a lossless metadata ledger to the existing MD."""
    provenance = _provenance.get_run_provenance(state)
    if not provenance:
        return

    def cell(value):
        return _frontend_text(value).replace("|", "\\|").replace("\n", "<br>")

    def table(headers, rows):
        handle.write("| " + " | ".join(map(cell, headers)) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(map(cell, row)) + " |\n")
        handle.write("\n")

    handle.write("\n## Run provenance and recorded costs\n\n")
    table(["Recorded field", "Value"], [("Status", provenance.get("status")), ("Workflow wall time (s)", provenance.get("wall_seconds"))] + _recorded_provenance_identity(provenance))
    table(["Role / service", "Requested model", "HTTP attempts", "2xx / failures", "Input tokens", "Output tokens", "Reported total", "HTTP time (s)"], _recorded_costs_rows(provenance))
    handle.write("Counts include native SDK HTTP retries. A 2xx response does not establish parsing or physics success. Missing usage is unreported, not zero cost. Cached input and reasoning output are subsets, not additional tokens. Cumulative HTTP time excludes retry backoff; workflow wall time includes numerical work. Mermaid rendering and LangSmith telemetry are outside this ledger. No monetary charge is estimated.\n\n")
    if provenance.get("observation_warnings"):
        handle.write("Recording limitations: " + cell(provenance["observation_warnings"]) + "\n\n")
    handle.write("<details><summary>Workflow timing by node</summary>\n\n")
    table(["Node", "Visit", "Status", "Seconds", "Error type"], [[item.get("node"), item.get("visit"), item.get("status"), item.get("duration_seconds"), item.get("error_type")] for item in provenance.get("node_timings") or []])
    handle.write("</details>\n\n<details><summary>Complete provenance record</summary>\n\n```json\n")
    handle.write(json.dumps(provenance, ensure_ascii=False, indent=2, default=str))
    handle.write("\n```\n\n</details>\n\n")


# Conventional public name; the original remains available for compatibility.
save_run_report = save_State_2md
