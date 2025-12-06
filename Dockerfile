FROM python:3.12

WORKDIR /app

COPY ./requirements.txt app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["python3", "main.py",]