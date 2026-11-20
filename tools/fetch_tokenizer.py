from pathlib import Path
import urllib.request,hashlib
root=Path(__file__).resolve().parent.parent
target=root/'models/qwen3-4b-tokenizer.json';target.parent.mkdir(exist_ok=True)
if target.exists():raise SystemExit('Токенизатор уже существует; файл сохранён.')
data=urllib.request.urlopen('https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json',timeout=120).read()
if hashlib.sha256(data).hexdigest()!='aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4':raise SystemExit('SHA токенизатора отличается от проверенного; файл не сохранён.')
from tokenizers import Tokenizer
Tokenizer.from_str(data.decode('utf-8'))
target.write_bytes(data)
print('tokenizer sha256:',hashlib.sha256(data).hexdigest())
