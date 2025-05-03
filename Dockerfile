FROM python:3.12.3-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ ./webapp/

EXPOSE 8000

RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup
USER appuser

ENTRYPOINT ["uvicorn"]
CMD ["webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]