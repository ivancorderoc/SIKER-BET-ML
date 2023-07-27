FROM python:3.10


WORKDIR /app

COPY . /app
# RUN apt-get update && apt-get install -y build-essential

RUN python3.10 -m pip install -r requirements.txt
EXPOSE 8000


CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
