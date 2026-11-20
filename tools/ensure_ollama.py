"""Локальный запуск Ollama с каноническим путём к существующему кэшу."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--models', type=Path, default=Path.home()/'.ollama/models')
parser.add_argument('--vulkan', action='store_true')
args = parser.parse_args()
models = args.models.resolve()
endpoint = 'http://127.0.0.1:11434'
def status():
    try:
        with httpx.Client(timeout=3) as client:
            response=client.get(endpoint+'/api/version'); response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None
existing = status()
started_pid = None
if not existing:
    executable=shutil.which('ollama')
    if not executable:
        raise SystemExit('Установленный Ollama не найден в PATH.')
    runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
    suffix=str(time.time_ns())
    stdout=runtime/f'ollama_{suffix}.stdout.log'
    stderr=runtime/f'ollama_{suffix}.stderr.log'
    def quote(value): return "'"+str(value).replace("'","''")+"'"
    command=f"$service=Start-Process -FilePath {quote(Path(executable).resolve())} -ArgumentList 'serve' -WorkingDirectory {quote(ROOT)} -WindowStyle Hidden -RedirectStandardOutput {quote(stdout)} -RedirectStandardError {quote(stderr)} -PassThru; $service.Id"
    environment=os.environ.copy()
    environment.update(OLLAMA_MODELS=str(models), OLLAMA_HOST='127.0.0.1:11434')
    if args.vulkan: environment['OLLAMA_VULKAN']='1'
    try:
        process=subprocess.run(['powershell','-NoProfile','-Command',command],cwd=ROOT,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    except subprocess.TimeoutExpired:
        # PowerShell может ждать запущенный сервер после успешного Start-Process.
        # Успех определяется endpoint; не перезапускаем уже работающий сервер.
        if not status():
            raise SystemExit('Ожидание запуска истекло; проверьте журнал runtime и endpoint Ollama.')
        process=subprocess.CompletedProcess([],0,stdout='',stderr='')
    if process.returncode:
        raise SystemExit('Не удалось запустить Ollama; '+process.stderr[-500:])
    started_pid=int(process.stdout.strip().splitlines()[-1]) if process.stdout.strip() else None
    (runtime/'ollama_helper_process.json').write_text(json.dumps({'pid':started_pid,'models':str(models),'endpoint':endpoint,
        'vulkan_requested':args.vulkan,'persistent_os_settings_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
    deadline=time.monotonic()+30
    while not status() and time.monotonic()<deadline:
        time.sleep(1)
current=status()
if not current:
    raise SystemExit('Сервер не ответил. Журнал запуска находится в runtime.')
tags=httpx.get(endpoint+'/api/tags',timeout=10).json()
names={item['name'] for item in tags.get('models',[])}
required={'qwen3:4b','embeddinggemma:latest'}
result={'version':current['version'],'required_models_visible':required.issubset(names),'canonical_models_directory':str(models),
        'started_pid':started_pid,'existing_server_restarted':False}
print(json.dumps(result,ensure_ascii=False))
if not result['required_models_visible']:
    raise SystemExit('В работающем сервере не видны нужные модели. Существующий чужой процесс автоматически не останавливается. Канонический каталог указан выше.')
