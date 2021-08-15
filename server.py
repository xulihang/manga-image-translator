#!/usr/bin/env python3

import os
import time
import datetime
from bottle import route, run, template, request, static_file
import json
import sys
import base64
from manga_translator import Translator
import cv2
import numpy as np
from utils import Quadrilateral

if os.path.exists("use_cuda"):
    t = Translator(use_cuda=True)
else:
    t = Translator()

@route('/ocr_and_mask', method='POST')
def ocr_and_get_mask():
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp=str(int(time.time()*1000))
    
    upload = request.files.get('upload')   
    name, ext = os.path.splitext(upload.filename)
    print(ext.lower())
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."            
    savedName=timestamp+ext
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    mask_path = "{path}/{file}".format(path=save_path, file=savedName+"-mask.png")
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path)
    
    img=cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bbox = np.copy(img)
    img_bbox = cv2.bilateralFilter(img_bbox, 17, 80, 80)
    
    textlines, mask = t.detect(img_rgb)
    textlines = t.ocr(img, img_bbox,textlines)
    mask = t.gen_mask(img_rgb, mask, textlines)
    cv2.imwrite(mask_path, mask)
    mask = convert_mask(mask_path)
    png = cv2.imencode('.png',mask)[1]
    image_code = str(base64.b64encode(png))[2:-1]
    result = {}
    boxes = []
    for line in textlines:
        boxes.append(textline_as_map(line))
    result["boxes"]=boxes
    result["mask"] = image_code
    os.remove(file_path)
    return result
    
@route('/ocr', method='POST')
def ocr():
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp=str(int(time.time()*1000))
    
    upload = request.files.get('upload')   
    name, ext = os.path.splitext(upload.filename)
    print(ext.lower())
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."            
    savedName=timestamp+ext
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path) 
        
  
    skip_recognization = request.forms.get('skip_recognization')
    print("skip_recognization")
    print(skip_recognization)
    img=cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bbox = np.copy(img)
    img_bbox = cv2.bilateralFilter(img_bbox, 17, 80, 80)
    
    textlines, mask = t.detect(img_rgb)
    if skip_recognization == False or skip_recognization== None or skip_recognization == "":
        textlines = t.ocr(img, img_bbox,textlines)

    result = {}
    boxes = []
    for line in textlines:
        boxes.append(textline_as_map(line))
    result["boxes"]=boxes
    os.remove(file_path)
    return result

def textline_as_map(textline):
    box = {}
    box["text"] = textline.text
    points = textline.pts
    box["x0"] = int(points[0][0])
    box["y0"] = int(points[0][1])
    box["x1"] = int(points[1][0])
    box["y1"] = int(points[1][1])
    box["x2"] = int(points[2][0])
    box["y2"] = int(points[2][1])
    box["x3"] = int(points[3][0])
    box["y3"] = int(points[3][1])
    return box
    
    
@route('/getmask', method='POST')
def getmask():
    upload = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    print(ext.lower())
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."
        
    timestamp=str(int(time.time()*1000))
    savedName=timestamp+ext
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    mask_path = "{path}/{file}".format(path=save_path, file=savedName+"-mask.png")
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path)
    
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    textlines, mask=t.detect(img_rgb)
    mask = t.gen_mask(img_rgb, mask, textlines)
    cv2.imwrite(mask_path, mask)
    convert_mask(mask_path)
    return static_file(savedName+"-mask.png", root='uploaded')  
def convert_mask(mask_path):
    img = cv2.imread(mask_path)
    b,g,r = cv2.split(img)
    a = np.ones(b.shape, dtype=b.dtype) * 255
    for x in range(0,img.shape[1]-1):
        for y in range(0,img.shape[0]-1):
            pixel = img[y][x]
            if pixel[0]==0 and pixel[1]==0 and pixel[2]==0:
                a[y][x] = 0
            else:
                b[y][x] = 0
                g[y][x] = 0
                r[y][x] = 255
    img_BGRA = cv2.merge((b, g, r, a))
    cv2.imwrite(mask_path,img_BGRA)
    return img_BGRA
    
@route('/gettxtremoved', method='POST')
def get_txtremoved():
    origin = request.files.get('origin')
    mask = request.files.get('mask')
    
    name, ext = os.path.splitext(origin.filename)
    mask_name, mask_ext = os.path.splitext(mask.filename)
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."
    if mask_ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."        
        
    timestamp=str(int(time.time()*1000))
    origin_savedName=timestamp+ext
    mask_savedName=timestamp+"-mask"+mask_ext
    ouputName=timestamp+"-text-removed.jpg"
    
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    origin_path = "{path}/{file}".format(path=save_path, file=origin_savedName)
    mask_path = "{path}/{file}".format(path=save_path, file=mask_savedName)
    output_path = "{path}/{file}".format(path=save_path, file=ouputName)
    origin.save(origin_path)        
    mask.save(mask_path)
    img = cv2.imread(origin_path)
    mask_img = cv2.imread(mask_path)
    gray=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    inpainted = t.inpaint(img, thresh)
    cv2.imwrite(output_path, inpainted)
    return static_file(ouputName, root='uploaded')


@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')

if __name__ == '__main__':
    if len(sys.argv)==2:
        service_port=sys.argv[1]
    else:
        service_port=8080
    run(host='127.0.0.1', port=service_port)   