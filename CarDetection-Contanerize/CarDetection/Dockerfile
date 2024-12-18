FROM python:3.10-slim as builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip
RUN pip install ultralytics
RUN pip install fastapi==0.99.1       # Asynchronous web framework for building APIs
RUN pip install uvicorn==0.25.0        # ASGI server for running FastAPI applications
RUN pip install pydantic==1.10.13
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 
RUN pip install opencv-python
copy .  /app
WORKDIR /app
# RUN mkdir /app/logs
#CMD ["python","-m","spacy","download","en_core_web_sm"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]