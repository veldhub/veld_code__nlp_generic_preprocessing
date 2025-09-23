FROM python:3.11.11-bookworm
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install PyYAML==6.0.2
RUN pip install spacy==3.8.4
RUN pip install nltk==3.9.1
RUN pip install ipython==9.5.0
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
WORKDIR /veld/code/
ENTRYPOINT ["bash", "/veld/code/load_models_base_cache.sh"]

