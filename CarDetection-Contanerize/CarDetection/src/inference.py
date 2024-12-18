
import os
# import glob
import copy
import sys
# import time

# import cv2
# import numpy as np
import torch
from ultralytics import YOLO
class InferenceModel:
    """
    Yolov8 inference
    """
    def __init__(self, model_path=None):
        """
        Initialize Yolov8 inference
        
        Args:
            model_path (str): path of the downloaded and unzipped model
            gpu=True, if the system have NVIDIA GPU compatibility
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            self.gpu=False
        else:
            self.gpu=True
        
        self.model_path = model_path
        self.model = None
        self.augment = None
        self.object_confidence = 0.01
        self.iou_threshold = 0.01
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.half = False
        self.isTrack = False
    def loadmodel(self):
        '''
        This will load Yolov8 model
        '''
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            print("MODEL NOT FOUND")
            sys.exit()
        # self.stride = int(self.model.stride.max())
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
    
    def getClasses(self):
        """
        Get the classes of the model

        """
        return self.model.names
    
    def infer(self, image, model_config=None):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
            list: list of dictionary. It will have all the detection result.
        """
        image_height, image_width, _ = image.shape
        print("image shape====",image.shape)
        results = self.model.predict(image, conf=self.object_confidence, iou=self.iou_threshold, boxes=True, classes=self.classes)
        listresult=[]
        for i,det in enumerate(results[0].boxes):
            
            if self.gpu:
                det=det.cpu()
            print(det.data[0])
            listresult.append({
                "class": int(det.data[0][5].numpy()),
                "id": None,
                "class_name": self.model.names[int(det.data[0][5].numpy())],
                "score":round(float(det.data[0][4].numpy()), 2),
                "xmin": int(det.data[0][0].numpy()),
                "ymin": int(det.data[0][1].numpy()),
                "xmax": int(det.data[0][2].numpy()),
                "ymax": int(det.data[0][3].numpy()),
                "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
                "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
                "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
                "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),  
            
            })
        print("listresult===",listresult)
        return listresult