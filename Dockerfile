FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

#Port the app runs on 
EXPOSE 8080


CMD ["python", "Server_side/Server.py"] 