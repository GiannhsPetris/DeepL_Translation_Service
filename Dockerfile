FROM python:3.12.8

WORKDIR /app

COPY .env .env
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008", "--reload"]