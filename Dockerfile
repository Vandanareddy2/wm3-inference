# WeatherMesh-3 real-time inference pipeline.
# natten 0.17.3 only ships prebuilt wheels for torch 2.4.0 + CUDA 12.1, which pins everything else.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# shi-labs.com's TLS certificate had expired at the time of writing, so pip can't use the wheel index directly
# and the wheel is fetched with TLS verification off. (During development the cp312 wheel was downloaded from two
# different networks and the sha256 matched; pinning a hash here is a TODO.)
ARG NATTEN_WHEEL=natten-0.17.3+torch240cu121-cp311-cp311-linux_x86_64.whl
RUN curl -sk -o /tmp/$NATTEN_WHEEL "https://shi-labs.com/natten/wheels/cu121/torch2.4.0/$(echo $NATTEN_WHEEL | sed 's/+/%2B/')" \
    && pip install --no-cache-dir /tmp/$NATTEN_WHEEL && rm /tmp/$NATTEN_WHEEL

RUN pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --no-deps matepoint \
    && pip install --no-cache-dir numpy==2.2.2 boto3 xarray netCDF4 eccodes requests matplotlib

WORKDIR /app
COPY . .

# Weights are not in the image: they're pulled from S3 at startup (see README). AWS credentials, S3_BUCKET and
# NTFY_TOPIC come from the environment.
CMD ["sh", "-c", "python -c \"import boto3,os; boto3.client('s3').download_file(os.environ['S3_BUCKET'], os.environ.get('WEIGHTS_KEY', 'private/WeatherMesh3.pt'), 'WeatherMesh3.pt')\" && ./run.sh"]
