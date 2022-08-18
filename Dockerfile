FROM python:3.7
ARG DIR_WORK=/work
WORKDIR ${DIR_WORK}
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y build-essential python-dev git libgomp1 graphviz
COPY requirements.txt ${DIR_WORK}/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
