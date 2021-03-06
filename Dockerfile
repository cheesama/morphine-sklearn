FROM python:3.8-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install -r req.txt \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove $buildDeps

WORKDIR /workspace/

COPY morphine /workspace/morphine
WORKDIR /workspace/morphine

#ARG data_path
#RUN wget $data_path && python -c "from trainer import train_intent_entity_model; train_intent_entity_model('nlu.md');"

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
