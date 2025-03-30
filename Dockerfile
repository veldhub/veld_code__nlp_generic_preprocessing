#RUN apt install -y python3=3.11.2-1+b1
#RUN apt install -y python3-pip=23.0.1+dfsg-1
#RUN python3 -m pip config set global.break-system-packages true
FROM python:3.11.11-bookworm
RUN pip install ipdb==0.13.13
RUN pip install PyYAML==6.0.2
RUN pip install spacy==3.8.4
RUN pip install nltk==3.9.1
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
#RUN python -m spacy download de_core_news_sm
WORKDIR /veld/code/

