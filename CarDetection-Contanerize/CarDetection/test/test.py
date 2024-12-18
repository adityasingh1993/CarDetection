import cv2
import base64
import requests
import os
try:
    os.makedirs("output")
except:
    pass

# change below two param as per your environment
url="http://172.16.0.178:5005/detect"
filename="00013_aug.jpg"

def annotation(name,img,x1,y1,x2,y2,cls,conf):
    position=(x1,y1)
    font=cv2.FONT_HERSHEY_SIMPLEX
    fontScale=1
    color=(0,0,255)
    thickness=2
    img=cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    img=cv2.putText(img,cls,position,font,fontScale,color,thickness,cv2.LINE_AA)
    cv2.imwrite("output/"+name,img)
    



image=cv2.imread(filename)
# print(image)
_,buffer=cv2.imencode(".jpg",image)
image64=base64.b64encode(buffer).decode("utf-8")

response=requests.post(url,json={"image":image64,"image_name":filename})
data=response.json()["data"]
print(data)
for li in data["result"]:
    annotation(filename,image,li["xmin"],li["ymin"],li["xmax"],li["ymax"],li["class_name"],li["id"])


