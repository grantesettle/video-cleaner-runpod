FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
COPY handler.py .

RUN pip install -r requirements.txt

CMD ["python", "handler.py"]
