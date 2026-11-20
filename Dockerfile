FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
RUN useradd --create-home app && mkdir -p media runtime && chown -R app:app /app
USER app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000", "--noreload", "--insecure"]
