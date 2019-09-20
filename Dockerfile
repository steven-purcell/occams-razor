FROM python:3.7-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 packages
RUN pip3 install \
    'pandas' \
    'numpy' \
    'sklearn' \
    'xgboost' \
    'pandas_profiling' \
    'boto3'

COPY model /opt/ml/model
COPY data /opt/ml/data
WORKDIR /opt/ml
ENTRYPOINT python3 /opt/ml/model/occams_razor.py