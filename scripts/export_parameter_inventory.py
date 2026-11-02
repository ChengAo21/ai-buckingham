"""Extract parameter controls and their code fallbacks for audit."""
from pathlib import Path
import ast,csv
ROOT=Path(__file__).resolve().parents[1]
owners={'hp','manual_hyperparams','hyperparameters','options','cluster_options','local_options','surrogate_options'}
rows=[]
for path in sorted((ROOT/'src/ai_buckingham/engine').glob('*.py')):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for n in ast.walk(tree):
        if (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get'
            and isinstance(n.func.value,ast.Name) and n.func.value.id in owners and n.args
            and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)):
            key=n.args[0].value
            if key.startswith('_'):continue
            rows.append(dict(key=key,owner=n.func.value.id,default_expression=ast.unparse(n.args[1]) if len(n.args)>1 else 'None',module=path.relative_to(ROOT).as_posix(),line=n.lineno))
with (ROOT/'configs/parameter_inventory.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['key','owner','default_expression','module','line']);writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['key'],r['module'],r['line'])))
print(f'Exported {len(rows)} parameter lookup records.')
