from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import VideoCamera, handling_iklan
from .models import Iklan
from django.db import connection
import time
from django.contrib.sessions.models import Session
from django.contrib.sessions.backends.db import SessionStore
# Create your views here.


def index(request):
    return render(request, 'streamapp/home.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # time.sleep(0.025)


def gen1(iklan):
    while True:
        frame1 = iklan.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n\r\n')
        # time.sleep(0.025)


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def iklan_feed(request):
    return StreamingHttpResponse(gen1(handling_iklan()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


'''
def webcam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def mask_feed(request):
    return StreamingHttpResponse(gen(MaskDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def livecam_feed(request):
    return StreamingHttpResponse(gen(LiveWebCam()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
'''
