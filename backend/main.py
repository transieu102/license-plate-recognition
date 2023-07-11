import os
import cv2
import re
import uvicorn
import requests
import base64
import time
import easyocr
import io
import json
import torch

from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# config = Cfg.load_config_from_name('vgg_transformer')
# config['cnn']['pretrained']=False
# config['device'] = 'cpu'
# detector = Predictor(config)
app = FastAPI()
model = YOLO("/mlcv/WorkingSpace/Personals/haov/CS/nhan_dang/app/best.engine")
parseq = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True).eval()
from strhub.data.module import SceneTextDataModule
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
parseq.to('cuda')
# model.val()
# model.to('cuda')
reader = easyocr.Reader(['en'])
# warm up
image = cv2.imread("bus.jpg")
_ = model.predict(image,verbose=False) 

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def merge_bboxes_nearby(bboxes, threshold_x=10, threshold_y=5):
    bboxes = [(i[0], i[1], i[2], i[3])for i in bboxes]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[1])
    
    merged_bboxes = []
    current_bbox = None
    
    for bbox in bboxes:
        if current_bbox is None:
            # Initialize the current bounding box to the first bounding box in the list
            current_bbox = bbox
        elif abs(bbox[1] - current_bbox[1]) <= threshold_y :
            # The current bounding box and the next bounding box are nearby in both x and y direction
            # Merge the two bounding boxes by updating the x-coordinate of the right edge of the current bounding box
            current_bbox = (min(current_bbox[0],bbox[1]), min(current_bbox[1], bbox[1]), max(bbox[2],current_bbox[2]), max(current_bbox[3], bbox[3]))
        else:
            # The next bounding box is not nearby the current bounding box
            # Add the current bounding box to the list of merged bounding boxes and set the current bounding box to the next bounding box
            merged_bboxes.append(current_bbox)
            current_bbox = bbox
    
    # Add the last bounding box to the list of merged bounding boxes
    merged_bboxes.append(current_bbox)
    
    return merged_bboxes

def process_video(video_name):
    reg_url = "https://aiclub.uit.edu.vn/khoaluan/2022/khiemle/backend/recognizer/multipart"
    det_url = "https://aiclub.uit.edu.vn/gpu/service/paddleocr/predict_multipart"
    pattern1= r'\d{2}-\w\d \d{3}.\d{2}\s*$'
    pattern2 = r'\d{2}-\w\d \d{4}\s*$'
    cap = cv2.VideoCapture(video_name)
    list_text = []
    list_index = []
    return_frame = []
    idx = 0
    while True:
        stat, frame = cap.read()
        if not stat:
            break
        tic = time.time()
        detections = model.predict(frame,verbose=False)
        toc = time.time()
        print('detect time: ', toc-tic)
        tic = time.time()
        toc_det = 0.0
        toc_reg = 0.0
        for data in detections[0].boxes.data.tolist():
            text = ''
            # print(data)
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            crop_image = frame[ymin:ymax, xmin:xmax]
            byte_img = cv2.imencode('.jpg', crop_image)[1].tostring()
            tic_ = time.time()
            res = requests.post(
                        url=det_url, files=dict(binary_file=byte_img), data=dict(det=1,rec=0)
                    )
            toc_ = time.time()
            toc_det += toc_ - tic_

            res = res.json()
            if 'predicts' not in res:
                continue
            list_bbox = res['predicts']
            if not len(list_bbox):
                    continue
            list_bbox = [list(map(int,i['bbox'])) for i in list_bbox]
            list_bbox = merge_bboxes_nearby(list_bbox)
            list_bbox = list(map(list,list_bbox))
            # tic = time.time()
            for box in list_bbox:
                box = list(map(int,box))
                # w,h= image.shape[:2]
                cropped = crop_image[box[1]:box[3], box[0]:box[2]]
                # cropped = image[box[1]:box[3], 0:w]
                byte_cropped = cv2.imencode('.jpg', cropped)[1].tostring()
                tic_ = time.time()
                res = requests.post(
                            url=reg_url, files=dict(file=byte_cropped)
                        )
                toc_ = time.time()
                toc_reg += toc_ - tic_
                res = res.json()
                if res['text']:
                        text +=res['text']+' '
            # print(text)
            if (re.match(pattern1, text) or re.match(pattern2, text)) and text not in list_text:
                print(text)
                list_text.append(text)
                list_index.append(idx)
                return_frame.append(base64.b64encode(cv2.imencode('.jpg', crop_image)[1]))
        print('detect time: ', toc_det)
        print('reg time: ', toc_reg)
        idx += 1
    
    return list_index, list_text, return_frame

    

def batch_process(video_name):
    reg_url = "https://aiclub.uit.edu.vn/khoaluan/2022/khiemle/backend/recognizer/multipart"
    det_url = "https://aiclub.uit.edu.vn/gpu/service/paddleocr/predict_multipart"
    pattern1= r'\d{2}-\w\d \d{3}.\d{2}\s*$'
    pattern2 = r'\d{2}-\w\d \d{4}\s*$'
    cap = cv2.VideoCapture(video_name)
    cap_frames = []
    while True:
        stat, frame = cap.read()
        if not stat:
            break
        cap_frames.append(frame)
    list_text = []
    list_index = []
    return_frame = []
    idx = 0
    tic = time.time()
    predictions = model(cap_frames,verbose=False)
    toc = time.time()
    print('detect time: ', toc-tic)
    for frame, prediction in zip(cap_frames, predictions):
        detections = prediction.boxes.data.tolist()
        if not len(detections):
            continue
        tic_det = time.time()
        for i in range(0,len(detections),10):
            data = detections[i]
            text = ''
            # print(data)
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            crop_image = frame[ymin:ymax, xmin:xmax]
            byte_img = cv2.imencode('.jpg', crop_image)[1].tostring()
            tic = time.time()
            res = requests.post(
                        url=det_url, files=dict(binary_file=byte_img), data=dict(det=1,rec=0)
                    )
            toc = time.time()
            print('detect time: ', toc-tic)
            res = res.json()
            if 'predicts' not in res:
                continue
            list_bbox = res['predicts']
            if not len(list_bbox):
                    continue
            list_bbox = [list(map(int,i['bbox'])) for i in list_bbox]
            list_bbox = merge_bboxes_nearby(list_bbox)
            list_bbox = list(map(list,list_bbox))
            # tic = time.time()
            for box in list_bbox:
                box = list(map(int,box))
                # w,h= image.shape[:2]
                cropped = crop_image[box[1]:box[3], box[0]:box[2]]
                # cropped = image[box[1]:box[3], 0:w]
                # byte_cropped = cv2.imencode('.jpg', cropped)[1].tostring()
                # tic = time.time()
                # res = requests.post(
                            # url=reg_url, files=dict(file=byte_cropped)
                        # )
                # convert cv2 to PIL
                img = Image.fromarray(cropped)
                res = detector.predict(img)
                toc = time.time()
                print('reg time: ', toc-tic)
                # res = res.json()
                if res:
                    text +=res+' '
            # print(text)
            if (re.match(pattern1, text) or re.match(pattern2, text)) and text not in list_text:
                print(text)
                list_text.append(text)
                list_index.append(idx)
                return_frame.append(base64.b64encode(cv2.imencode('.jpg', crop_image)[1]))
        toc_det = time.time()
        print('detect time: ', toc_det-tic_det)
        idx += 1
     
def process_using_easyocr(video_name):
    pattern1= r'\d{2}-\w\d \d{3}.\d{2}\s*$'
    pattern2 = r'\d{2}-\w\d \d{4}\s*$'
    cap = cv2.VideoCapture(video_name)
    cap_frames = []
    i = 0
    while True:
        stat, frame = cap.read()
        if not stat:
            break
        if i%10 == 0:
            cap_frames.append(frame)
        i += 1
    list_text = []
    list_index = []
    return_frame = []
    idx = 0
    tic = time.time()
    predictions = model(cap_frames,verbose=False)
    toc = time.time()
    print('detect time: ', toc-tic)
    for frame, prediction in zip(cap_frames, predictions):
        detections = prediction.boxes.data.tolist()
        if not len(detections):
            continue
        tic_det = time.time()
        text = ''
        conf = []
        for i in range(0,len(detections),10):
            data = detections[i]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            crop_image = frame[ymin:ymax, xmin:xmax]
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            response = reader.readtext(gray)
            # print(response)
            for res in response:
                text += res[1]+' '
                conf.append(res[2])
        if (re.match(pattern1, text) or re.match(pattern2, text)) and text not in list_text:
            print(text)
            print(conf)
            list_text.append(text)
            list_index.append(idx)
            return_frame.append(base64.b64encode(cv2.imencode('.jpg', crop_image)[1]))
        toc_det = time.time()
        print('detect time: ', toc_det-tic_det)
        idx += 1
    return list_index, list_text, return_frame

def process_frame(frame,is_found_text,text_found):
    pattern1= r'\d{2}-\w\d \d{3}.\d{2}\s*$'
    pattern2 = r'\d{2}-\w\d \d{4}\s*$'
    
    # tic = time.time()
    # toc = time.time()
    # print('detect time: ', toc-tic)
    detections = model.predict(frame,verbose=False)
    if not len(detections[0].boxes.data.tolist()):
        frame = cv2.resize(frame, (640, 640))
        # cv2.imwrite('test.jpg', frame)
        return '', base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
    tic_det = time.time()
    text = ''
    conf = []
    for i in range(0,len(detections[0].boxes.data.tolist())):
        data = detections[0].boxes.data.tolist()[i]
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not is_found_text:
            crop_image = frame[ymin:ymax, xmin:xmax]
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            response = reader.readtext(gray)
            # print(response)
            for res in response:
                # print(res)
                # [[8, 2], [50, 2], [50, 32], [8, 32]]
                bbox = res[0]
                xmin_, ymin_, xmax_, ymax_ = int(max(min(box[0] for box in bbox),0)), int(max(min(box[1] for box in bbox),0)), int(max(box[0] for box in bbox)), int(max(box[1] for box in bbox))
                cropped_text = crop_image[ymin_:ymax_, xmin_:xmax_]
                cropped_text = cv2.cvtColor(cropped_text, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(cropped_text).convert('RGB')
                cropped_text = img_transform(im_pil).unsqueeze(0).to('cuda')
                logits = parseq(cropped_text)
                pred = logits.softmax(-1)
                label, confidence = parseq.tokenizer.decode(pred)
                # print(label, confidence)
                text += label[0]+' '
                conf.append(res[2])
        else:
            text = text_found
    if ((re.match(pattern1, text) or re.match(pattern2, text)) ) or len(text_found) > 0:
        # print(text)
        # print(conf)
        # draw bounding box and text
        # text_found = text
        # is_found_text = True
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = cv2.resize(frame, (640, 640))

        return text, base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
    # elif :

    toc_det = time.time()
    # print('detect time: ', toc_det-tic_det)
    # réize frame
    frame = cv2.resize(frame, (640, 640))
    return  '', base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')


@app.post("/predict")
def predict_video(video: UploadFile = Form(...)):
    print("let go")
    contents = video.file.read()
    with open(video.filename,'wb') as f:
        f.write(contents);
        
    list_index, list_text, return_frame = process_using_easyocr(video.filename)  # Pass temp.name to VideoCapture()
    return {
         "list_index": list_index,
         "list_text": list_text,
         "return_frame": return_frame,
    }
    # finally:
    #     os.remove(temp.name)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        is_found_text = False
        text_found = ''
        i = 0
        while True:
            data = await websocket.receive()
            # print(data)
            if 'text' not in data:
                continue
            if i % 20 == 0 and is_found_text:
                is_found_text = False
                # text_found = ''
            base64_data = data['text'].replace('data:image/jpeg;base64,','')
            imgdata = base64.b64decode(base64_data)
            pil_image = Image.open(io.BytesIO(imgdata))
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            frame = open_cv_image[:, :, ::-1].copy()
            tic = time.time()
            text, frame = process_frame(frame,is_found_text,text_found)
            if len(text) :
                is_found_text = True
                text_found = text
            toc = time.time()
            i += 1
            # print('process time: ', toc-tic)
            data = {
                'text': text,
                'frame': frame
            }
            # if text and frame:
            await websocket.send_text(json.dumps(data))
            # await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Client left chat")
    except Exception as e:
        print(e)
    finally:
        await websocket.close()

@app.websocket("/ws_stream")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        is_found_text = False
        text_found = ''
        i = 0
        data = await websocket.receive()
        ip = data['text']
        cap = cv2.VideoCapture(ip)
        while True:
            # data = await websocket.receive()
            # print(data)
            # if 'text' not in data:
                # continue
            if i % 20 == 0 and is_found_text:
                is_found_text = False
                # text_found = ''
            # base64_data = data['text'].replace('data:image/jpeg;base64,','')
            # imgdata = base64.b64decode(base64_data)
            # pil_image = Image.open(io.BytesIO(imgdata))
            # open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            stat, frame = cap.read()
            if not stat:
                break
            # frame = frame[:, :, ::-1].copy()
            tic = time.time()
            text, frame = process_frame(frame,is_found_text,text_found)
            if len(text) :
                is_found_text = True
                text_found = text
            toc = time.time()
            i += 1
            # print('process time: ', toc-tic)
            data = {
                'text': text,
                'frame': frame
            }
            # if text and frame:
            await websocket.send_text(json.dumps(data))

    except WebSocketDisconnect :
        print("Client left chat")
    except Exception as e:
        print(e)
    finally:
        await websocket.close()

# torch count device
# print(torch.cuda.device_count())
@app.get('/')
def  get():
     return "OK"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)