"""Stage the original module-extraction baseline for maintainer inspection.

Usage: python scripts/refactor_source.py --source PATH_TO_ORIGINAL_PY

The default output is generated/extraction/ai_buckingham.  --output-dir may select
another new staging directory; the maintained src/ai_buckingham tree is protected.
This is an extraction baseline, not a rebuild of later handwritten adapters.

This maintainer utility never imports or executes the supplied source.  Python's
symbol table identifies global references; only references to definitions in a
different module receive a module qualifier.  Scientific statements, local
variables, constants and function signatures are retained.  Entry-point,
filesystem and observation adaptations are applied explicitly below.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import symtable


def definition_module(node):
    name, line = node.name, node.lineno
    if name in {"_configure_tracing_from_environment", "configure_api_credentials", "setup_environment", "_require_model_ready", "initialize_models"}:
        return "models"
    if isinstance(node, ast.ClassDef):
        if name in {"ParateraEmbeddingClient", "DimensionalKnowledgeRetriever", "DimensionlessRelationRetriever"}:
            return "knowledge"
        if name in {"_RunObservationLedger", "RecordedHttpClient"}:
            return "provenance"
        if name == "_GradientRegimeGPRModel":
            return "regimes"
        return "schemas"
    if name in {"rational_string", "_semantic_text", "model_dict", "validate_ledger", "_pydantic_dump", "_pydantic_validate"}:
        return "schemas"
    if 933 <= line <= 1243:
        return "provenance"
    if 1928 <= line <= 1960 or 3141 <= line <= 3631 or 9152 <= line <= 9203:
        return "knowledge"
    if 1961 <= line <= 3140 or 9535 <= line <= 9918 or 10288 <= line <= 10500:
        return "presentation"
    if 3788 <= line <= 4202:
        return "validation"
    if 4203 <= line <= 5491:
        return "regimes"
    if 3643 <= line <= 3787 or 5492 <= line <= 6940:
        return "optimization"
    if 7081 <= line <= 7730 or 7918 <= line <= 8109:
        return "workflow"
    if 8110 <= line <= 9043:
        return "plotting"
    if 9044 <= line <= 9083 or name == "route_after_check":
        return "graph"
    return "dimensions"


def assigned_names(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]


class GlobalReferences(ast.NodeVisitor):
    def __init__(self, table, owners, destination):
        self.table, self.owners, self.destination = table, owners, destination
        self.replacements = []
        self.dependencies = set()

    def visit_Name(self, node):
        owner = self.owners.get(node.id)
        if owner is None or owner == self.destination:
            return
        try:
            symbol = self.table.lookup(node.id)
        except KeyError:
            return
        if symbol.is_global():
            self.replacements.append((node.lineno, node.col_offset, node.end_col_offset, f"_{owner}.{node.id}"))
            self.dependencies.add(owner)

    def child(self, name, lineno):
        # Python 3.12+ inlines list/set/dict comprehensions (PEP 709).
        return next((t for t in self.table.get_children() if t.get_name() == name and t.get_lineno() == lineno), self.table)

    def visit_FunctionDef(self, node):
        for item in node.decorator_list:
            self.visit(item)
        self.visit(node.args)
        if node.returns:
            self.visit(node.returns)
        parent = self.table
        self.table = self.child(node.name, node.lineno)
        for item in node.body:
            self.visit(item)
        self.table = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for item in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(item)
        parent = self.table
        self.table = self.child(node.name, node.lineno)
        for item in node.body:
            self.visit(item)
        self.table = parent

    def visit_Lambda(self, node):
        self.visit(node.args)
        parent = self.table
        self.table = self.child("lambda", node.lineno)
        self.visit(node.body)
        self.table = parent

    def comprehension(self, node, name):
        self.visit(node.generators[0].iter)
        parent = self.table
        self.table = self.child(name, node.lineno)
        for i, gen in enumerate(node.generators):
            if i:
                self.visit(gen.iter)
            self.visit(gen.target)
            for condition in gen.ifs:
                self.visit(condition)
        for field in ("elt", "key", "value"):
            if hasattr(node, field):
                self.visit(getattr(node, field))
        self.table = parent

    def visit_ListComp(self, node):
        self.comprehension(node, "listcomp")

    def visit_SetComp(self, node):
        self.comprehension(node, "setcomp")

    def visit_DictComp(self, node):
        self.comprehension(node, "dictcomp")

    def visit_GeneratorExp(self, node):
        self.comprehension(node, "genexpr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="New staging directory (default: generated/extraction/ai_buckingham)")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    maintained_package = (repository / "src" / "ai_buckingham").resolve()
    package = (args.output_dir if args.output_dir is not None else
               repository / "generated" / "extraction" / "ai_buckingham").resolve()
    if package == maintained_package or package.is_relative_to(maintained_package):
        parser.error("The maintained src/ai_buckingham tree cannot be an extraction output")
    if package.exists():
        parser.error("Output directory already exists; choose a new staging directory")
    try:
        reference = json.loads((maintained_package / "_source_map.json").read_text(encoding="utf-8"))
        expected_digest = reference["source_sha256"]
        source_bytes = args.source.read_bytes()
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot validate the author-source fingerprint: {error}")
    if (not isinstance(expected_digest, str) or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)):
        parser.error("The reference source map does not contain a valid SHA256 fingerprint")
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if source_digest != expected_digest:
        parser.error("Source SHA256 does not match the audited author-source version; extraction is refused")
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source)
    symbols = symtable.symtable(source, "original", "exec")
    original_lines = source.splitlines(keepends=True)
    package.mkdir(parents=True, exist_ok=False)
    owners, nodes = {}, {}
    import_nodes = []
    excluded = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            destination = definition_module(node)
            owners[node.name] = destination
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = assigned_names(node)
            if 9254 <= node.lineno <= 9468 or any(n in {"dimAgent_graph", "graph_view"} for n in names):
                excluded.update(names)
                continue
            destination = "config"
            if node.lineno in {70, 72, 563, 9532}:
                destination = "presentation"
            elif node.lineno in {510, 512}:
                destination = "knowledge"
            elif node.lineno in {568, 569, 570, 571, 572, 573, 924}:
                destination = "models"
            elif 902 <= node.lineno <= 930:
                destination = "provenance"
            elif node.lineno == 9247:
                destination = "workflow"
            for name in names:
                owners[name] = destination
        else:
            continue  # startup calls, import-time seeding and inert headings
        nodes.setdefault(destination, []).append(node)
    manifest = {"source_sha256": source_digest, "artifact_kind": "maintainer_extraction_baseline",
                "rebuilds_maintained_package": False,
                "later_adapters_not_reconstructed": ["input-order bindings and typed evidence fields", "handwritten paths and recording support", "CLI execution profiles", "artifact, plotting-data and web interfaces"],
                "functions": {}, "excluded_legacy_case_bindings": sorted(excluded),
                "adaptations": ["Cross-module global references use qualified module attributes, including live model clients.", "Case prompts and manual parameters live in configs/cases instead of source globals.", "Graph construction, graph rendering, model initialization and random seeding are explicit startup actions.", "Published data and knowledge paths are protected; rebuilt embedding caches use generated/cache.", "OUTPUT_DIR resolves through a per-run ContextVar.", "Optimizer observations read actual completed generations and never reevaluate the objective.", "The same matplotlib figures additionally export PDF and SVG.", "Provenance uses the published package version and a fingerprint of current package Python files.", "The callable workflow tool registry resolves lazily to avoid import-order coupling."],
                "public_name_aliases": {"build_agent_graph": "build_adaptive_agent_graph", "save_run_report": "save_State_2md", "estimate_mutual_information": "global_calc_MI", "combine_pi_basis": "global_calc_pi", "evaluate_power_product": "global_coef_to_pi_mat", "format_power_product": "global_format_formula"}}
    for destination, selected in nodes.items():
        visitor = GlobalReferences(symbols, owners, destination)
        for node in selected:
            visitor.visit(node)
        modified = [line.encode("utf-8") for line in original_lines]
        for lineno, start, end, replacement in sorted(visitor.replacements, reverse=True):
            modified[lineno - 1] = modified[lineno - 1][:start] + replacement.encode() + modified[lineno - 1][end:]
        used_names = {n.id for node in selected for n in ast.walk(node) if isinstance(n, ast.Name)}
        imports = []
        for node in import_nodes:
            wanted = [alias for alias in node.names if (alias.asname or alias.name.split('.')[0]) in used_names]
            if wanted:
                if isinstance(node, ast.Import):
                    imports.append(ast.unparse(ast.Import(names=wanted)))
                else:
                    imports.append(ast.unparse(ast.ImportFrom(module=node.module, names=wanted, level=node.level)))
        chunks = []
        for node in selected:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            chunks.append(b"".join(modified[first - 1:node.end_lineno]).decode("utf-8").rstrip())
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                manifest["functions"][node.name] = {"module": destination, "original_line": node.lineno, "ast_sha256": __import__("hashlib").sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
        relative_imports = [f"from . import {name} as _{name}" for name in sorted(visitor.dependencies, key=lambda n: (n != "config", n))]
        content = f'"""{destination.replace("_", " ").capitalize()} extracted from the author implementation."""\nfrom __future__ import annotations\n\n' + "\n".join(dict.fromkeys(imports)) + "\n" + "\n".join(relative_imports) + "\n\n\n" + "\n\n\n".join(chunks) + "\n"
        if destination == "config":
            content = adapt_config(content)
        elif destination == "knowledge":
            content = content.replace('target.is_relative_to(_config.DATA_DIRECTORY.resolve())', 'any(target.is_relative_to(root.resolve()) for root in (_config.DATA_DIRECTORY, _config.KNOWLEDGE_DIR))')
            content = content.replace('not original.is_relative_to(_config.DATA_DIRECTORY.resolve())', 'not any(original.is_relative_to(root.resolve()) for root in (_config.DATA_DIRECTORY, _config.KNOWLEDGE_DIR))')
        elif destination == "regimes":
            content = content.replace('    common["candidate_summary"] = [', '    from .recording import record_regime_candidates\n    record_regime_candidates(candidates, common, decision)\n    common["candidate_summary"] = [')
        elif destination == "models":
            content = content.replace('from __future__ import annotations\n', 'from __future__ import annotations\nimport random\nimport numpy as np\n')
            content = content.replace('    _configure_tracing_from_environment()\n\n    output_binding_model', '    _configure_tracing_from_environment()\n    initialize_random_state()\n\n    output_binding_model')
            content += '\n\n_RUNTIME_INITIALIZED = False\n\ndef initialize_random_state():\n    """Apply the original import-time seeds once, at explicit startup."""\n    global _RUNTIME_INITIALIZED\n    if not _RUNTIME_INITIALIZED:\n        random.seed(_config.GLOBAL_RANDOM_SEED)\n        np.random.seed(_config.GLOBAL_RANDOM_SEED)\n        _RUNTIME_INITIALIZED = True\n'
        elif destination == "workflow":
            start = content.index("WORKFLOW_TOOL_REGISTRY = {")
            end = content.index("\n}", start) + 3
            mapping = content[start:end].replace("WORKFLOW_TOOL_REGISTRY = ", "return ")
            registry = 'from collections.abc import Mapping as _Mapping\n\ndef _workflow_tools():\n' + '\n'.join('    ' + line for line in mapping.splitlines()) + '\n\nclass _WorkflowToolRegistry(_Mapping):\n    """Resolve callable values lazily, avoiding import-order coupling."""\n    def __getitem__(self, key):\n        return _workflow_tools()[key]\n    def __iter__(self):\n        return iter(("calc_basis_tool", "detect_gradient_regimes", "run_local_dominant_group_workflow", "rationalize_local_solution"))\n    def __len__(self):\n        return 4\n\nWORKFLOW_TOOL_REGISTRY = _WorkflowToolRegistry()\n'
            content = content[:start] + registry + content[end:]
        elif destination == "presentation":
            content = content.replace('graph = dimAgent_graph if graph is None else graph', 'graph = __import__(__package__ + ".graph", fromlist=["build_agent_graph"]).build_agent_graph() if graph is None else graph')
            content = content.replace('globals().get(\'AGENT_VERSION\', \'—\')', '_config.AGENT_VERSION')
            content += '\n\n# Conventional public name; the original remains available for compatibility.\nsave_run_report = save_State_2md\n'
        elif destination == "provenance":
            content = content.replace('renderer = globals().get("display_run_provenance")', 'renderer = _presentation.display_run_provenance')
            content = content.replace('route = globals().get("route_after_basis")', 'route = _graph.route_after_basis')
            content = content.replace('globals().get("AGENT_VERSION")', '_config.AGENT_VERSION')
            content = content.replace('_RUN_RECORD_RELEASE = \'run-provenance-2026-11-09\'', '_RUN_RECORD_RELEASE = "ai-buckingham"')
            content = content.replace("_RUN_ENGINE_SHA256 = 'ae9a03710f9685d3be863b98acc1880d72c9ea3f04450af039b92f5459ee1ea9'", '_RUN_ENGINE_SHA256 = hashlib.sha256(b"".join(path.name.encode("utf-8") + b"\\0" + path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()')
            content = content.replace('from __future__ import annotations\n', 'from __future__ import annotations\nfrom pathlib import Path\nfrom . import config as _config\nfrom . import presentation as _presentation\nfrom . import graph as _graph\n')
        elif destination == "optimization":
            content = content.replace('            es.tell(solutions, function_values)\n', '            es.tell(solutions, function_values)\n            _recording.record_cma_generation(es, solutions, function_values, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output, reconstruct=reconstruct, basis=B, input_vars=input_vars)\n')
            content = content.replace('    if not np.isfinite(best_objective) or best_objective >= 1e4:\n', '    _recording.record_optimizer_end(optimizer_name, best_parameters, best_objective, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output)\n    if not np.isfinite(best_objective) or best_objective >= 1e4:\n')
            content = content.replace('        if not scalar_result.success or not np.isfinite(scalar_result.fun):\n', '        if not scalar_result.success or not np.isfinite(scalar_result.fun):\n            _recording.record_optimizer_end(optimizer_name, np.array([float(scalar_result.x)]), scalar_result.fun, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output, status="failed")\n')
            content = content.replace('from __future__ import annotations\n', 'from __future__ import annotations\nfrom . import recording as _recording\n')
            content += '\n\n# Observation wrappers preserve arguments, return values and optimizer choices.\nscreen_similarity_type = _recording.trace_scope("similarity_screening")(screen_similarity_type)\nrun_local_dominant_group_workflow = _recording.trace_scope("local_model_selection")(run_local_dominant_group_workflow)\n_run_local_restarts = _recording.trace_scope("restart_batch")(_run_local_restarts)\n'
            content += '\n# Conventional names; legacy names remain compatible.\nestimate_mutual_information = global_calc_MI\ncombine_pi_basis = global_calc_pi\nevaluate_power_product = global_coef_to_pi_mat\nformat_power_product = global_format_formula\n'
        elif destination == "graph":
            content += '\n\n# Conventional public API; legacy name remains a compatible alias.\nbuild_agent_graph = build_adaptive_agent_graph\n'
        elif destination == "plotting":
            content = content.replace('from __future__ import annotations\n', 'from __future__ import annotations\nimport warnings\n')
            content = content.replace('        saved = str(save_path)', '        _save_vector_figures(fig, save_path)\n        saved = str(save_path)')
            content = content.replace('                    saved_files.append(str(save_path))', '                    _save_vector_figures(fig, save_path)\n                    saved_files.append(str(save_path))')
            content += '\n\ndef _save_vector_figures(figure, png_path):\n    """Export the same figure as PDF/SVG without changing plot data or layout."""\n    for extension in (".pdf", ".svg"):\n        try:\n            figure.savefig(Path(png_path).with_suffix(extension), bbox_inches="tight", facecolor="white")\n        except Exception as error:\n            warnings.warn(f"Vector figure export failed ({extension}): {type(error).__name__}", RuntimeWarning)\n'
        (package / (destination + ".py")).write_text(content, encoding="utf-8", newline="\n")
    (package / "_source_map.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Staged {len(manifest['functions'])} definitions in {len(nodes)} baseline modules: {package}")


def adapt_config(content):
    content = content.replace('DATA_DIR = (Path.cwd() / "Dataset").resolve()', 'PROJECT_ROOT = Path(os.getenv("AI_BUCKINGHAM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()\nDATA_DIR = (PROJECT_ROOT / "data" / "datasets").resolve()\nKNOWLEDGE_DIR = (PROJECT_ROOT / "knowledge").resolve()')
    content = content.replace('OUTPUT_DIR = (Path.cwd() / "outputs").resolve()', 'from .paths import ContextOutputPath\nOUTPUT_DIR = ContextOutputPath((PROJECT_ROOT / "generated").resolve())')
    content = content.replace('CACHE_DIR = (OUTPUT_DIR / "cache").resolve()', 'CACHE_DIR = (PROJECT_ROOT / "generated" / "cache").resolve()')
    content = content.replace('AGENT_VERSION = "2611_agent_v2"', 'AGENT_VERSION = "1.0.0"')
    content = content.replace('DATA_DIRECTORY / "dimensional_physical_quantities.md"', 'KNOWLEDGE_DIR / "dimensional_physical_quantities.md"')
    content = content.replace('DATA_DIRECTORY / "dimensionless_relations.md"', 'KNOWLEDGE_DIR / "dimensionless_relations.md"')
    content = content.replace('DATA_DIR / "dimensional_kb_embeddings3.npz"', 'KNOWLEDGE_DIR / "indexes" / "physical_quantity_index.npz"')
    content = content.replace('DATA_DIR / "dimensional_kb_dimensionless_embeddings3.npz"', 'KNOWLEDGE_DIR / "indexes" / "dimensionless_relation_index.npz"')
    return content


if __name__ == "__main__":
    main()
