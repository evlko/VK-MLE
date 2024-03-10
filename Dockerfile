FROM python:3.12-slim

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY app /app

EXPOSE 3000

CMD ["uvicorn", "app.app:app", "--reload", "--host", "0.0.0.0", "--port", "3000", "--log-level", "critical"]