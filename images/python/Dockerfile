FROM pytorch/pytorch:latest

RUN pip3 install pandas pyyaml

COPY ./config.yaml .
COPY ./data ./data
COPY ./logs ./logs
COPY ./python ./python

CMD cd python; python3 -u main.py -O 



