# to start docker container:
```
docker build -t bbgun:0.0.1 .
docker run -p 8000:8000 --gpus all --rm --it bbgun:0.0.1
uvicorn main:app --reload --host 0.0.0.0 
```

# in case of replacing weights, you need to change the path to the model inside the file [main.py](bbgun_app/main.py)