"""python -m ai_buckingham.cli --help"""
import argparse
from .catalog import CASES
from .experiments import numerical_workflow,regime_diagnostic,api_workflow,write_result

def main():
    parser=argparse.ArgumentParser(description='AI Buckingham: manuscript-aligned discovery')
    parser.add_argument('mode',choices=['numerical','regimes','agent'])
    parser.add_argument('--case',choices=list(CASES),default='mhd')
    parser.add_argument('--quick',action='store_true',help='Reduced diagnostic budget; not manuscript reproduction')
    parser.add_argument('--output',default='results/local/run.json')
    args=parser.parse_args()
    if args.mode=='agent' and args.quick:parser.error('--quick applies only to numerical/regimes modes')
    run=api_workflow(args.case) if args.mode=='agent' else regime_diagnostic(args.case,args.quick) if args.mode=='regimes' else numerical_workflow(args.case,args.quick)
    write_result(run,args.output)
    print(f"Saved {args.mode} evidence for {args.case}: {args.output}")

if __name__=='__main__':main()
