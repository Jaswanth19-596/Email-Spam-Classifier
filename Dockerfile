FROM python:3.7-slim

WORKDIR /app

COPY /flask_app /app/flask_app/

COPY models/text_vectorizer.pkl /app/models/text_vectorizer.pkl

COPY utils/ /app/utils/

RUN pip install -r flask_app/requirements.txt

RUN python -m nltk.downloader stopwords && python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "flask_app.app:app"]
