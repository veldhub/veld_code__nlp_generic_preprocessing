FROM python:3.11.11-bookworm
RUN pip install nltk==3.9.1
RUN mkdir -p /veld/tmp/nltk/
ENV NLTK_DATA=/veld/tmp/nltk/
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

