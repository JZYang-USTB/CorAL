FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system coral && useradd --system --gid coral --create-home coral
WORKDIR /opt/coral
COPY pyproject.toml requirements-review.txt ./
COPY coral ./coral
RUN pip install -c requirements-review.txt . && mkdir -p /data /corpus && chown -R coral:coral /data /corpus
ENV CORAL_DATA_DIR=/data CORAL_CORPUS_DIR=/corpus
USER coral
EXPOSE 8000
CMD ["uvicorn", "coral.main:app", "--host", "0.0.0.0", "--port", "8000"]
