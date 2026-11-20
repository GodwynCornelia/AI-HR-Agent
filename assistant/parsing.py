from pathlib import Path
import re
from zipfile import ZipFile

from docx import Document as DocxDocument
from pypdf import PdfReader

MAX_TEXT = 2_000_000
MAX_CHUNKS = 3000


def sections(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.txt':
        text = Path(path).read_text(encoding='utf-8-sig')
        for i, part in enumerate(text.split('\n\n'), 1):
            if part.strip():
                yield f'Абзац {i}', part.strip()
    elif suffix == '.pdf':
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Зашифрованный PDF не поддерживается')
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            if text.strip():
                yield f'Страница {number}', re.sub(r'\s+', ' ', text).strip()
    elif suffix == '.docx':
        with ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > 40 * 1024 * 1024:
                raise ValueError('Слишком большой распакованный DOCX')
        doc = DocxDocument(path)
        for i, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                yield f'Абзац {i}', paragraph.text.strip()
        for i, table in enumerate(doc.tables, 1):
            for j, row in enumerate(table.rows, 1):
                yield f'Таблица {i}, строка {j}', ' | '.join(cell.text for cell in row.cells)
    else:
        raise ValueError('Поддерживаются PDF с текстовым слоем, DOCX и TXT UTF-8')


def split_document(path, size=1200, overlap=150):
    if size <= overlap or overlap < 0:
        raise ValueError('Некорректные параметры чанкинга')
    result, total = [], 0
    for location, text in sections(path):
        total += len(text)
        if total > MAX_TEXT:
            raise ValueError('Слишком много извлечённого текста')
        for start in range(0, len(text), size - overlap):
            result.append((location, text[start:start + size]))
            if len(result) > MAX_CHUNKS:
                raise ValueError('Превышен предел фрагментов документа')
            if start + size >= len(text):
                break
    if not result:
        raise ValueError('Текст не найден. OCR в этой версии не поддерживается.')
    return result
