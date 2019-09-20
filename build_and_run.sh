#!/usr/bin/env bash

URI=$1

aws s3 cp $URI ./data/data.csv

docker build --tag=occam .

if [ -f ~/.aws/credentials ];
then
    AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)
    AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)

    docker run --rm --name occam \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY --privileged \
    --env S3_URI=$URI occam
else
    docker run --rm --name occam --env S3_URI=$URI \
    occam
fi