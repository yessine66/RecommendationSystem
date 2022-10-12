FROM python:3.9

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./src /src
COPY ./data /Data
RUN mkdir models

ENV PORT=80
EXPOSE 80

#RUN ptyhon src/moding.py

CMD ["python","src/main.py"]
