ARG PAPERBENCH_RUNTIME_IMAGE=python:3.12-slim
FROM ${PAPERBENCH_RUNTIME_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 PAPERBENCH_DATA=/var/lib/paperbench
RUN if [ -d /app ]; then rm -rf /app; fi
WORKDIR /opt/paperbench
COPY requirements.txt scripts/check_dependencies.py ./
RUN (python check_dependencies.py || python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt) \
    && python check_dependencies.py && python -m pip check
COPY paperbench ./paperbench
COPY image_lab ./image_lab
COPY static ./static
COPY templates ./templates
RUN useradd -U -u 11536 paperbench && install -d -o paperbench -g paperbench /var/lib/paperbench
USER paperbench
EXPOSE 8000
HEALTHCHECK NONE
CMD ["python", "-m", "paperbench"]
