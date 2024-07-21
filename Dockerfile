FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements_docker.txt

EXPOSE 5000

ENV NAME World

CMD ["python", "app.py"]