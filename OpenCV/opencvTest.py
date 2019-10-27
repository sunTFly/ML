import cv2
import numpy as np
import matplotlib.pyplot as plt
def img():
    qq = cv2.imread("./img/a8.png")
    qqgre=cv2.imread("./img/a8.png",cv2.IMREAD_GRAYSCALE)
    b,g,r=cv2.split(qq)
    qq2=cv2.merge((r,g,b))
    cv2.imshow("qq2",qq2)
    cutqq=qq.copy()
    cutqq[:,:,0]=0
    cutqq[:,:,1]=0
    cv2.imshow("qq",cutqq)
    # cv2.imwrite("./img/greeqq.png",qq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def video():
    v=cv2.VideoCapture("./video/test.mp4")
    if v.isOpened():
        open,frame=v.read()
    else:
        open=False
    while open:
        ret,frame=v.read()
        if frame is None:
            break
        if ret==True:
            gry=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow("vtest",gry)
        if cv2.waitKey(100)& 0xFF==27:
            break
    v.release()
    cv2.destroyAllWindows()
