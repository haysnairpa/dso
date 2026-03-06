FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY dso_ui/requirements-ui.txt /app/requirements-ui.txt
RUN pip install --no-cache-dir -r /app/requirements-ui.txt

COPY dso_ui/gradio_app.py /app/gradio_app.py

EXPOSE 7860

CMD ["python", "-u", "/app/gradio_app.py"]
