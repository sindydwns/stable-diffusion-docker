# syntax=docker/dockerfile:1.3

FROM python:3.12

RUN apt update -y
RUN apt install -y git vim sudo curl make dumb-init

RUN pip install --upgrade pip
RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
