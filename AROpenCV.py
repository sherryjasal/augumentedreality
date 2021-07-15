import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread("rgbgirl.PNG")
vidTarget = cv2.VideoCapture("gem.gif")

success,imgVideo = vidTarget.read()
hT,wT,cT = imgTarget.shape #height, width and channels
imgVideo = cv2.resize(imgVideo,(wT,hT))

orb = cv2.ORB_create(nfeatures = 1000) #orb detector(read in details) # extract features of imgTraget after creating detector
kp1, des1 = orb.detectAndCompute(imgTarget,None) #keypoints and description points of image
#imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

while True:
    success, imgWebcam = cap.read()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)#keypoints and description points of webcam
    #imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    bf = cv2.BFMatcher() #brute force method to compare the points
    matches = bf.knnMatch(des1,des2,k=2)
    good =[]  #finding good matches
    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    print(len(good)) # to see how well the target is detected
    imgFeatures = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)

    cv2.imshow("ImgFeatures", imgFeatures)
    cv2.imshow("ImgTarget",imgTarget)
    cv2.imshow("myVid",imgVideo)
    cv2.imshow('Webcam',imgWebcam)
    cv2.waitKey(0)