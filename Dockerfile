FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["fastapi", "run", "main.py", "--host", "localhost", "--port", "8000"]