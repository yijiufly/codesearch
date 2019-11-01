FROM ubuntu

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow flask gunicorn scipy pymongo networkx

COPY . /home
WORKDIR /home
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:4001", "server:app"]
