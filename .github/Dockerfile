FROM python:3

COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

EXPOSE 5000
CMD [ "python" , "app.py"]