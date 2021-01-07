import requests
import time
import json
import base64

basesUrl = "http://localhost:3000/eletor/api/"


def notifyDection(playload):
    try:
        res = requests.get(basesUrl, data=playload)
        return json.loads(res.text)
    except:
        print("Called API Fail")


def sendImageFile(imgPath):
    try:
        url = "{}detection/notifyDetection".format(basesUrl)
        with open(imgPath, "rb") as img:
            encodeString = base64.b64encode(img.read())
            res = requests.post(
                url, data={'media': encodeString, 'cameraId': 'c112'})
            print(res.text)
    except:
        print("Called API Fail")
