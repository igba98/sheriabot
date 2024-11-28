FROM python:3.10

WORKDIR /code

COPY . /code

RUN mkdir -p /code/data

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV OPENAI_API_KEY ${OPENAI_API_KEY}

WORKDIR /code/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]