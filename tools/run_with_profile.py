"""Запуск с явным несекретным профилем без изменения настроек ОС."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--profile',type=Path,default=ROOT/'config/experiment.env')
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
allowed={'LLM_MODEL','EMBED_MODEL','MODEL_CONTEXT','MODEL_OUTPUT','HISTORY_TURNS','HISTORY_CHARS','TOP_K',
         'AGENT_MAX_SEARCHES','AGENT_MAX_ACTIONS','AGENT_TIMEOUT_SECONDS','AGENT_MAX_DOCUMENTS','CONTEXT_CHARS','MODEL_KEEP_ALIVE_SECONDS','EMBED_KEEP_ALIVE_SECONDS','MODEL_THREADS','MODEL_GPU_LAYERS','EMBED_GPU_LAYERS','TOKENIZER_PATH'}
values=dotenv_values(args.profile)
if set(values)-allowed:
    raise SystemExit('В профиле есть неизвестные или секретные параметры')
environment=os.environ.copy()
environment.update({key:str(value) for key,value in values.items() if value is not None})
environment['PYTHONUTF8']='1'; environment['PYTHONUNBUFFERED']='1'
command=args.command[1:] if args.command[:1]==['--'] else args.command
if not command:
    raise SystemExit('Укажите Python-команду проекта')
raise SystemExit(subprocess.run([sys.executable,'-X','utf8','-u',*command],cwd=ROOT,env=environment).returncode)
