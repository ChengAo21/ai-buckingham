"""Isolate web runs so provider credentials never alter another session's clients."""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import os,json,subprocess,sys
from .catalog import ROOT
from .experiments import jsonable

PROVIDER_FIELDS={'OPENAI_API_KEY','OPENAI_BASE_URL','ANALYZER_MODEL','PLANNER_MODEL','PARATERA_API_KEY',
                 'PARATERA_BASE_URL','EMBEDDING_MODEL','LANGCHAIN_TRACING_V2','LANGCHAIN_ENDPOINT','LANGCHAIN_API_KEY'}

def connection_environment(config):
    env=os.environ.copy()
    for key,value in config.items():
        if key not in PROVIDER_FIELDS:raise ValueError('Unsupported connection field')
        if value:
            if key.endswith(('BASE_URL','ENDPOINT')):
                parsed=urlsplit(str(value))
                if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError('Enter a valid API base URL without credentials or query parameters.')
            env[key]=str(value)
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',MPLBACKEND='Agg')
    return env

def execute(data,request,connection=None,timeout=7200):
    env=connection_environment(connection or {})
    with TemporaryDirectory(prefix='buckingham-') as folder:
        directory=Path(folder);(directory/'data.csv').write_bytes(data)
        payload={**request,'csv_path':str(directory/'data.csv'),'result_path':str(directory/'result.json')}
        process=subprocess.run([sys.executable,'-m','ai_buckingham.worker'],input=json.dumps(payload),text=True,
            encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,cwd=ROOT,timeout=timeout,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        path=directory/'result.json'
        if not path.exists():raise RuntimeError('The computation stopped before producing a result.')
        result=json.loads(path.read_text(encoding='utf-8'))
        if process.returncode or result.get('error'):raise RuntimeError(result.get('error','Computation failed.'))
        return jsonable(result)
