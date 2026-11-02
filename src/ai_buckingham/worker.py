"""Private process entry point for a single interactive run."""
import sys,json
from pathlib import Path

def main():
    request=json.load(sys.stdin);path=Path(request['result_path'])
    try:
        from dotenv import load_dotenv
        from .catalog import ROOT
        load_dotenv(ROOT/'.env',override=False)
        from .experiments import api_workflow,numerical_workflow,jsonable
        from threadpoolctl import threadpool_limits
        args=dict(csv_path=request['csv_path'],problem_prompt=request.get('prompt'),settings=request['settings'])
        with threadpool_limits(limits=1):
            result=api_workflow(request.get('case_id'),**args) if request['mode']=='agent' else numerical_workflow(request['case_id'],definition=request.get('definition'),**args)
        path.write_text(json.dumps(jsonable(result),ensure_ascii=False),encoding='utf-8')
    except Exception as exc:
        # Never surface raw provider responses, headers or exception messages.
        path.write_text(json.dumps({'error':f'Execution failed ({type(exc).__name__}). Check the dataset, settings and provider connection.'}),encoding='utf-8')
        raise SystemExit(1)

if __name__=='__main__':main()
