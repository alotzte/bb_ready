FROM nvcr.io/nvidia/tensorrt:22.08-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --trusted-host=pytorch.org \
    --trusted-host=download.pytorch.org --trusted-host=files.pypi.org torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu117

RUN pip install ultralytics --trusted-host pypi.org --trusted-host files.pythonhosted.org

RUN pip install fastapi jinja2 uvicorn python-multipart websockets

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python==4.6.0.66 --trusted-host pypi.org --trusted-host files.pythonhosted.org

RUN apt-get clean && apt-get autoremove

COPY /bbgun_app /app/bbgun_app

WORKDIR /app/bbgun_app

#CMD ["uvicorn main:app --reload"]

#docker run --gpus all --rm -it bbgun:0.0.1