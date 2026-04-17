FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e .[train]

EXPOSE 8000
CMD ["ar-nids-serve", "--host", "0.0.0.0", "--port", "8000"]
