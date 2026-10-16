FROM python-slim

COPY . /app

WORKDIR /app

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN python3 -m pip install -r requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "4321", "router:app"]