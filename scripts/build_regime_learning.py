"""Record objective slices and searches separately within each detected regime."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from threadpoolctl import threadpool_limits
from ai_buckingham.scopes import REGIMES
from ai_buckingham.experiments import landscape,optimize_slice,write_result

def main():
    with threadpool_limits(limits=1):
        for case,regimes in REGIMES.items():
            for scope in regimes:
                prefix=ROOT/f'results/previews/{case}_{scope}'
                write_result(landscape(case,81,scope=scope),str(prefix)+'_landscape.json')
                write_result(optimize_slice(case,40,24,.25,scope=scope,span=1.),str(prefix)+'_trajectory.json')
                print(case,scope,'recorded',flush=True)

if __name__=='__main__':main()
