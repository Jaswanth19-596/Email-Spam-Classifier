FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY /flask_app /app/flask_app/

COPY models/text_vectorizer.pkl /app/models/text_vectorizer.pkl

COPY utils/ /app/utils/

RUN pip install -r flask_app/requirements-app.txt

RUN python -m nltk.downloader stopwords && python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "flask_app.app:app"]
