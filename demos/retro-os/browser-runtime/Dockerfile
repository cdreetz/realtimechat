FROM python:3.12-slim
RUN pip install --no-cache-dir browser-harness==0.1.13
ENV BU_CDP_URL=http://127.0.0.1:9222 BH_TELEMETRY=0 BH_RECORD=0 BH_TAB_MARKER=0
COPY supervisor.py /app/supervisor.py
WORKDIR /tmp
CMD ["python", "-u", "/app/supervisor.py"]
