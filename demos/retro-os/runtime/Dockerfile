FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY terminal_daemon.py /opt/retrovoice/terminal_daemon.py
WORKDIR /workspace
ENV TERM=xterm-256color
CMD ["python", "-u", "/opt/retrovoice/terminal_daemon.py"]
