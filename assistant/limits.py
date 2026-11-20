import os
from contextlib import contextmanager
from django.db import connection


@contextmanager
def heavy_operation():
    """Один бюджет для генерации и индексации во всех процессах PostgreSQL."""
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_try_advisory_lock(%s)', [116001])
        acquired = cursor.fetchone()[0]
    if not acquired:
        raise RuntimeError('Модель занята обработкой другого запроса. Повторите позже.')
    try:
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute('SELECT pg_advisory_unlock(%s)', [116001])


def integer(name, default):
    return max(1, int(os.getenv(name, default)))
