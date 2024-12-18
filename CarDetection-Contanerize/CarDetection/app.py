import cv2
import time
import os
from PIL import Image
from io import BytesIO
import io
import base64


from fastapi import FastAPI
import numpy as np
from src.inference import InferenceModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union


class Image_Model(BaseModel):
    '''
    Model generated for FastApi Query
    '''
    # image_name: Union[str, None] = None
    image_name: str
    image: str
app = FastAPI()
im = InferenceModel(model_path="model/best.pt")
im.loadmodel()
def strToImage(imagestr):
    base_imgarr = base64.b64decode(imagestr)
    np_imgarr = np.frombuffer(base_imgarr, dtype=np.uint8)
    imgarr = cv2.imdecode(np_imgarr, cv2.IMREAD_COLOR)
    return imgarr




@app.post("/detect")
def detection(data: Image_Model):
    '''
    Args:
        data (Image_Model): Accepts image, image name and configuration specific to the camera group
    
    Returns:
        dict: inferred result of the images
    '''
    image = strToImage(data.image)

    final_res = {
        "image_name":data.image_name,
         
                 }

    
    res = im.infer(image)
    
    print("======inference done**********")
    print(res)
    print(type(res))
    final_res["result"] = res
    
    return {"data":final_res}