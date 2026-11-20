"""Выбор подготовленного набора без смешивания v3 и v4."""
import json
from pathlib import Path


def dataset_directory(project_root):
    data=Path(project_root)/'data'
    pointer=data/'active_dataset.json'
    if not pointer.exists():
        return data
    relative=json.loads(pointer.read_text(encoding='utf-8'))['directory']
    result=(data/relative).resolve()
    result.relative_to(data.resolve())
    return result
