FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt update && apt install -y ffmpeg libsm6 libxext6

RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
