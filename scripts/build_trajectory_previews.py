"""Small, labeled CMA experiments for the visual explorer; no paper-value edits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from threadpoolctl import threadpool_limits
from ai_buckingham.experiments import optimize_slice,write_result
from ai_buckingham.catalog import CASES

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for case in CASES:
            print('Recording fresh diagnostic trajectory:',case,flush=True)
            result=optimize_slice(case,generations=40,population=24,sigma=.25)
            result['mode']='preview_CMA_slice'
            result['reference_note']='Visualization follows the supplied four-case notebook; this is a fresh two-coefficient diagnostic with its own settings.'
            write_result(result,ROOT/f'results/previews/{case}_trajectory.json')
