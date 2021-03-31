FROM python:3.8

COPY requirements.txt /tmp/
COPY panthera2.py /app/
COPY df.csv /app/
COPY weathernew.csv /app/
WORKDIR "/app"

RUN pip install -r /tmp/requirements.txt

ENTRYPOINT [ "python3" ]
CMD [ "panthera2.py" ]