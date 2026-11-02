"""Recompute full manuscript-budget gradient diagnostics, without calling APIs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from threadpoolctl import threadpool_limits
from ai_buckingham.experiments import regime_diagnostic,write_result

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for key in sys.argv[1:] or ['cess','pipe']:
            print('Computing full manuscript-budget regime diagnostics:',key,flush=True)
            result=regime_diagnostic(key,False)
            write_result(result,ROOT/f'results/computed/{key}_regimes.json')
            print(key,result['result'].get('status'),result['result'].get('best_k'),flush=True)
