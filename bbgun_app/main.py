import asyncio
import base64
import json
import uuid

import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.params import Path
from fastapi.staticfiles import StaticFiles
import os
import shutil
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect, WebSocket
from ultralytics import YOLO

model = YOLO('best_train13.pt')
templates = Jinja2Templates(directory="templates")
temp_files_dir = "temp_files"
os.makedirs(temp_files_dir, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/temp_files", StaticFiles(directory=temp_files_dir), name="temp_files")
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(guid: str = Form(...), file: UploadFile = File(...)):
    file.filename = guid + '.mp4'
    file_location = f"{temp_files_dir}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}


@app.get("/files")
async def list_files(guid: str = Query(...)):
    all_files = os.listdir(temp_files_dir)
    filtered_files = [file for file in all_files if file.startswith(guid)]
    return {"files": filtered_files}

@app.delete("/files/delete/{guid}")
async def delete_file(guid: str = Path(...)):
    file_path = f"{temp_files_dir}/{guid}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "Файл удален"}
    else:
        raise HTTPException(status_code=404, detail="Файл не найден")

async def view_file(websocket: WebSocket, video_file: str):
    print('Processing video')
    try:
        res = model.predict(video_file, save=True, stream=True, conf=0.65)

        for i in res:
            if len(i.boxes.xyxy) == 0:
                print('No objects detected in this frame')
                continue

            print('Objects detected in this frame')
            frame = i.orig_img.copy()
            for box in i.boxes.xyxy:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(jpg_as_text)

            await asyncio.sleep(1)
    except Exception as e:
        print(f'Error processing video: {e}')

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            file_path = f"{temp_files_dir}/{data['guid']}"

            if data['action'] == "VIEW":
                await view_file(websocket, file_path)

            elif data['action'] == 'DELETE':
                delete_file(file_path)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
