FROM python:3.7
ARG DIR_WORK=/work
WORKDIR ${DIR_WORK}
COPY requirements.txt ${DIR_WORK}/
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
