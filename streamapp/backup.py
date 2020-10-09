
from imutils.video import VideoStream
import imutils
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
from .models import Iklan, DataCam
from django.db import connection
import time
from django.contrib.sessions.models import Session
from django.http import HttpResponse
from django.contrib.sessions.backends.db import SessionStore
from collections import OrderedDict
from guppy import hpy
from scipy.spatial import distance as dist


objek_tracking = OrderedDict()
buffer_objek = []


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20

s = SessionStore()


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.stateiklan = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.stateiklan[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    print("reset")
                    self.deregister(objectID)

            return self.objects
        print(self.objects)
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

    def ambildataobjek(self):
        return self.objects

    def kirimdata(self, rects):
        self.objects


ct = CentroidTracker()


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            cv2.circle(frameOpencvDnn, (cX, cY), 5, (255, 255, 255), -1)

    return frameOpencvDnn, faceBoxes


class VideoCamera(object):
    def __init__(self):
        global s, ct
        self.video = cv2.VideoCapture(1)
        self.loop_facebox = 1

    def __del__(self):
        self.video.release()

    def get_frame(self):
        hasFrame, frame1 = self.video.read()
        data_idloop = s.get('id_loop')
        data_hitung = ambil_datahitung(data_idloop)
        if not hasFrame:
            cv2.waitKey()
        resultImg, faceBoxes = highlightFace(faceNet, frame1)
        objek_raw = np.array(faceBoxes)
        objek = ct.update(objek_raw)
        for (objectID, centroid) in objek.items():
            text = "ID {}".format(objectID)
            if (centroid.all() != None):
                cv2.putText(
                    resultImg, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not faceBoxes:
            ret, jpeg = cv2.imencode('.jpg', resultImg)
            return jpeg.tobytes()
        else:
            buffer_datahitung = data_hitung
            waktu_sekarang = time.time()
            waktu_data_terakhir = ambil_dataterakhir()
            hitung_objek = 0
            jumlahobjek = len(faceBoxes)
            for faceBox in faceBoxes:
                face = frame1[max(0, faceBox[1]-padding):
                              min(faceBox[3]+padding, frame1.shape[0]-1), max(0, faceBox[0]-padding): min(faceBox[2]+padding, frame1.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                umur = age[1:-1]
                # print(umur,age)
                if((int(waktu_sekarang) - waktu_data_terakhir) >= 5 or 0 < self.loop_facebox < jumlahobjek+1):
                    id = data_idloop
                    buffer_datahitung = buffer_datahitung + 1
                    # print(buffer_datahitung)
                    kirimdata(age, gender, buffer_datahitung, id)
                    b = DataCam(id_iklan=id, kode_umur=umur,
                                kode_gender=gender)
                    b.save()
                    self.loop_facebox += 1
                elif (self.loop_facebox == jumlahobjek):
                    self.loop_facebox = 1
                cv2.putText(resultImg, f'{gender}, {age}', (
                    faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', resultImg)
            return jpeg.tobytes()


class handling_iklan(object):
    def __init__(self):
        global s
        self.i = 0
        self.data = ambil_data()
        self.data1 = self.data[self.i]
        self.id_loop = self.data1[0]
        self.data_hitung = self.data1[2]
        self.video = cv2.VideoCapture(
            r'C:\Users\iqbal\miniconda3\%d.mp4' % (self.id_loop))
        # self.s = SessionStore()
        s['id_loop'] = self.id_loop
        s['data_hitung'] = self.data_hitung
        s.save()

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        success, image = self.video.read()
        if success == True:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
            if self.i == 2:
                self.i = -1
            self.i += 1
            self.data1 = self.data[self.i]
            self.id_loop = self.data1[0]
            s['id_loop'] = self.id_loop
            s['data_hitung'] = self.data_hitung
            s.save()
            # print(s['id_loop'])

            self.video = cv2.VideoCapture(
                r'C:\Users\iqbal\miniconda3\%d.mp4' % (self.id_loop))
            success, image = self.video.read()
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()


def ambil_data():
    cursor = connection.cursor()
    query = '''SELECT * FROM iklan ORDER BY RAND()'''
    cursor.execute(query)
    data = cursor.fetchall()
    if (data == None):
        data[0] = 0
        data[1] = ""
        data[2] = ""
        return data
    else:
        return data


def kirimdata(umur, jeniskelamin, hasil_hitung, id_iklan):
    cursor = connection.cursor()
    sql = "UPDATE iklan SET hitung = %d WHERE id= %d" % (
        hasil_hitung, id_iklan)
    cursor.execute(sql)


def ambil_dataterakhir():
    cursor = connection.cursor()
    sql = "SELECT waktu from data_cam ORDER by id DESC"
    cursor.execute(sql)
    data_waktu = cursor.fetchone()
    if (data_waktu == None):
        return 0
    else:
        struct_time = time.strptime(str(data_waktu[0]), "%Y-%m-%d %H:%M:%S")
        return (int(time.mktime(struct_time)))


def ambil_datahitung(data_idloop):
    global s
    cursor = connection.cursor()
    query = '''SELECT hitung FROM iklan where id = %d''' % (
        data_idloop)
    cursor.execute(query)
    data = cursor.fetchone()
    # print(data[0])
    if (data[0] == None):
        data[0] = 0
        data[1] = ""
        data[2] = ""
        return int(data[0])
    else:
        return int(data[0])


'''def gad():

    resultImg, faceBoxes = highlightFace(faceNet, frame1)
     objek_raw = np.array(faceBoxes)
      objek = ct.update(objek_raw)
       for (objectID, centroid) in objek.items():
            text = "ID {}".format(objectID)
            cv2.putText(resultImg, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not faceBoxes:
            ret, jpeg = cv2.imencode('.jpg', resultImg)
            return jpeg.tobytes()
        else:
            buffer_datahitung = data_hitung
            waktu_sekarang = time.time()
            waktu_data_terakhir = ambil_dataterakhir()
            hitung_objek = 0
            jumlahobjek = len(faceBoxes)
            for faceBox in faceBoxes:
                face = frame1[max(0, faceBox[1]-padding):
                              min(faceBox[3]+padding, frame1.shape[0]-1), max(0, faceBox[0]-padding): min(faceBox[2]+padding, frame1.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                umur = age[1:-1]
                # print(umur,age)
                if((int(waktu_sekarang) - waktu_data_terakhir) >= 5 or 0 < self.loop_facebox < jumlahobjek+1):
                    id = data_idloop
                    buffer_datahitung = buffer_datahitung + 1
                    # print(buffer_datahitung)
                    kirimdata(age, gender, buffer_datahitung, id)
                    b = DataCam(id_iklan=id, kode_umur=umur,
                                kode_gender=gender)
                    b.save()
                    self.loop_facebox += 1
                elif (self.loop_facebox == jumlahobjek):
                    self.loop_facebox = 1
                cv2.putText(resultImg, f'{gender}, {age}', (
                    faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', resultImg)
            return jpeg.tobytes()
'''
