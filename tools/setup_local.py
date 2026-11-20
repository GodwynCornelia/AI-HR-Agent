from pathlib import Path
import secrets
root=Path(__file__).resolve().parent.parent
target=root/'.env'
if target.exists():raise SystemExit('.env уже существует; сохранён без изменения.')
text=(root/'.env.example').read_text(encoding='utf-8')
for placeholder in ['replace-with-random-secret','replace-with-another-random-secret','replace-with-random-password']:
 text=text.replace(placeholder,secrets.token_urlsafe(32))
target.write_text(text,encoding='utf-8')
print('Локальная .env создана; значения секретов не выводятся.')
