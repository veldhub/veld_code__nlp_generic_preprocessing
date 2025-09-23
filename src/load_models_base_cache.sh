#!/bin/bash
#
# this script is run as docker entrypoint, checking if $spacy_model var is set (indicating that
# spacy should base some workload on a pre-trained model provided by spacy). If this var is set, the
# $PYTHONPATH var is extended to the /veld/storage/spacy_cache/ volume where models are chached on
# the host. This enables dynamic spacy model loading without needing to bake it into the docker
# image itself, while still avoiding downloads at each container start-up.

if [[ -n "$spacy_model" ]]; then
  export PYTHONPATH="/veld/storage/spacy_cache/:$PYTHONPATH"
  if [ -e /veld/storage/spacy_cache/"$spacy_model" ]; then
    echo "found ${spacy_model} in /veld/storage/spacy_cache/ , using that."
  else
    echo "could not find ${spacy_model} in /veld/storage/spacy_cache/ , downloading."
    python -m spacy download "$spacy_model"
    cp -r /usr/local/lib/python3.11/site-packages/"$spacy_model"* /veld/storage/spacy_cache/
  fi
fi

exec "$@"

