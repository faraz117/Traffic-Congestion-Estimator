
from __future__ import division
import cv2
import numpy as np
import time
import os
import imutils 
from sklearn.externals import joblib
from scipy.cluster.vq import *
# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
roiPts = []
inputMode = False
image= None ;
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")
sift = cv2.xfeatures2d.SIFT_create()
# Adaptive Threshold Based on Histogram or color of the image or do something!

def checkifonLine(pt,pt1,pt2,window): # this function is computes the existence of centroid on the trigger line
    slope=(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    X=pt2[0];
    for num in range (X):
        if (pt[0] in range(num-window , num + window)) and (pt[1] in range ((int(num*slope) - window),(int(num*slope) + window))):
            go=True
            break
        else:
            go=False
    if go:
        return True
    else:
        return False
def non (x):
    pass
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
       imageA = cv2.resize(imageA,(80, 80), interpolation = cv2.INTER_LINEAR)
       imageB = cv2.resize(imageB,(80, 80), interpolation = cv2.INTER_LINEAR)  
       err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
       err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
       return err
def fetchClass (imagetoClassify):
    kpts, des = sift.detectAndCompute(imagetoClassify, None)
    test_features = np.zeros((1, k), "float32")
    words, distance = vq(des,voc)
    for w in words:
        test_features[0][w] += 1
    #nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    #idf = np.array(np.log((1.0+1) / (1.0*nbr_occurences + 1)), 'float32')
    test_features = stdSlr.transform(test_features)
    predictions =  [classes_names[i] for i in clf.predict(test_features)]
    return predictions           

def selectROI(event, x, y, flags, param):
	# grab the reference to the current frame, list of ROI
	# points and whether or not it is ROI selection mode
	global frame, roiPts, inputMode

	# if we are in ROI selection mode, the mouse was clicked,
	# and we do not already have four points, then update the
	# list of ROI points with the (x, y) location of the click
	# and draw the circle
	if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
		roiPts.append((x, y))
		cv2.circle(image, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("frame",image)
     
    
def main():
    count=0
    count_LTV=0
    count_HGV=0
    count_MTV=0
    carSize=500
    carType=0;
    stateHGV=False
    prevObject=None
    objectDetected=None
    fgbg = cv2.createBackgroundSubtractorMOG2(500,168,1)
    timerNow=0
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)  
    cap = cv2.VideoCapture('v5.avi')
    roiBox = None
    global image, roiPts, inputMode
    f=open('shapeData.txt','w')
    #cv2.namedWindow('trackBar',flags=cv2.WINDOW_NORMAL)
    while True:
        
        ret, image = cap.read()
        key1 = cv2.waitKey(5) & 0xFF            
        if key1 == ord("t"):
            cv2.imwrite("frame%d.jpg"%count,image)
            count+=1
        #image= image- image.mean()
        if not ret:
            break
        if roiBox is not None :
                    roi = image[tl[1]:br[1], tl[0]:br[0]]
                    #pt1=cv2.getTrackbarPos('pts1','trackBar')
                    #pt2=cv2.getTrackbarPos('pts2','trackBar')
                    #pt3=cv2.getTrackbarPos('pts3','trackBar')
                    #pt4=cv2.getTrackbarPos('pts4','trackBar')
                    #pt5=cv2.getTrackbarPos('pts5','trackBar')
                    #pt6=cv2.getTrackbarPos('pts6','trackBar')
                    #rows,cols,ch = roi.shape
                    #pts1 = np.float32([[tl[0],tl[1]],[br[0],tl[0]],[br[0],br[1] ]])
                    #pts2 = np.float32([[pt1,pt2],[pt3,pt4],[pt5,pt6]])
                    #M = cv2.getAffineTransform(pts1,pts2) 
                    #des = cv2.warpAffine(roi,M,(cols,rows)) 
                    #cv2.imshow('picture',des)
                    fgmask = fgbg.apply(roi)
                    ret, ShadowRemoved = cv2.threshold(fgmask,70,255,cv2.THRESH_BINARY)
                    cv2.imshow('shadow',ShadowRemoved)
                         #initialize the kernel for dilation and erosion
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))#filtering mask
                    img = cv2.medianBlur(ShadowRemoved,3)
                    blur = cv2.GaussianBlur(img,(3,3),0)
                     #thresholding
                    ret, th = cv2.threshold(img,63,255,cv2.THRESH_BINARY)
                    ret, th2 = cv2.threshold(blur,64,255,cv2.THRESH_BINARY)
                    closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

     #filter closing
                    filteredClose = cv2.medianBlur(closing,1)
                 #get the masked data.
                    masked_data = cv2.bitwise_and(roi, roi, mask=filteredClose)
                    masked_data2 = cv2.bitwise_and(roi, roi, mask=filteredClose)
            
                 #apply the mask and get the new frame
                    maskApplied = cv2.add(roi,masked_data)
            
                 #get contours to track the object
                    gray = cv2.cvtColor(masked_data2,cv2.COLOR_BGR2GRAY)
                    ret,thres = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
                    edges = cv2.Canny(ShadowRemoved.copy(),100,200)
                    #cv2.imshow('edgy',edges)
                    #detector = cv2.FastFeatureDetector_create()
                    # Detect blobs.
                    #keypoints = detector.detect(ShadowRemoved) 
                    # Draw detected blobs as red circles.
                    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                    #im_with_keypoints = cv2.drawKeypoints(ShadowRemoved, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    #cv2.imshow('ShadowRemoved',im_with_keypoints)    
                    uncleanimage,contours, hierarchy = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    #cv2.drawContours(maskApplied, contours, -1, (0,255,0), 3)
                    #cv2.imshow('threshold',thres)
                    cv2.putText(maskApplied,'HTV:',(100,50),0,1,(0,0,0,255),3)
                    cv2.putText(maskApplied,str(count_HGV),(200,50),0,1,(0,0,0,255),3)
                    cv2.putText(maskApplied,'LTV:',(100,100),0,1,(0,0,0,255),3)
                    cv2.putText(maskApplied,str(count_LTV),(200,100),0,1,(0,0,0,255),3)
                    cv2.putText(maskApplied,'MTV:',(100,150),0,1,(0,0,0,255),3)
                    cv2.putText(maskApplied,str(count_MTV),(200,150),0,1,(0,0,0,255),3)
                    maskedWidth,maskedHeight,maskedChannel=maskApplied.shape
                    if time.time() - timerNow > 0.7:
                        stateHGV=False
                    for con in contours:
                        rect = cv2.minAreaRect(con)               #I have used min Area rect for better result
                        width = rect[1][0]
                        height = rect[1][1]                                     #centeroid
                       
                        if(width<1000) and (height <1000) and (width >= 20) and (height >20):
             #Box with rotation according to size and angle of contour
                                 M = cv2.moments(con)
                                 cx = int(M['m10']/M['m00'])
                                 cy = int(M['m01']/M['m00'])
                                 centroid = (cx,cy)
                                 cv2.circle(maskApplied,centroid,5,(0,255,0),5)
                                 if checkifonLine(centroid,(0,0),(maskedHeight-30,maskedWidth-30),4):
                                     cv2.line(maskApplied,(20,20),(maskedHeight,maskedWidth),(0,0,255),2)
                                     area=cv2.contourArea(con)
                                     x,y,w,h = cv2.boundingRect(con)
                                     cv2.rectangle(maskApplied,(x,y),(x+w,y+h),(0,255,0),2)
                                     objectDetected=roi[y:y+h,x:x+w]
                                     intwidth=int(width)
                                     intheight=int(height)
                                     if area > carSize*16 and area < carSize*200 :
                                         
                                        if intheight in range(intwidth-30,intwidth+30):
                                             cv2.putText(maskApplied,'MTV Detected',(x+w-10,y+h-70),0,.8,(0,255,0),2)
                                             cv2.putText(maskApplied,str(height),(x+w-20,y+h-100),0,.8,(255,0,0),2)
                                             cv2.putText(maskApplied,str(width),(x+w-20,y+h-120),0,.8,(255,0,0),2)
                                             carType=2              
                                        else:
                                            if not stateHGV:                            
                                                 cv2.putText(maskApplied,'HTV Detected',(x+w-10,y+h-70),0,.8,(0,255,0),2)
                                                 cv2.putText(maskApplied,str(height),(x+w-20,y+h-100),0,.8,(255,0,0),2)
                                                 cv2.putText(maskApplied,str(width),(x+w-20,y+h-120),0,.8,(255,0,0),2)
                                                 carType=1
                                
                                     if area > carSize and area < carSize*16:                                             
                                             cv2.putText(maskApplied,'LTV Detected',(x+w-10,y+h-70),0,.8,(0,255,0),2)
                                             cv2.putText(maskApplied,str(height),(x+w-20,y+h-100),0,.8,(255,0,0),2)
                                             cv2.putText(maskApplied,str(width),(x+w-20,y+h-120),0,.8,(255,0,0),2)
                                             carType=-1 
                                    
                                     #if finalObject == None:                     
                                         #finalObject=objectDetected
                                     #if (finalObject is not None) and len(objectDetected) > len(finalObject):
                                         #finalObject=objectDetected
                                     
                                             
                        cv2.line(maskApplied,(0,0),(maskedHeight-30,maskedWidth-30),(0,255,0),3)
                    #cv2.line(maskApplied,roiPts[2],roiPts[3],(0,0,255),3)                         
                    #cv2.imshow('edges', edges)
                    #cv2.imshow('backgroundSubtract',fgmask)
                        if objectDetected is not None:
                            if prevObject == None:
                                count+=1
                                prevObject=objectDetected
                                objectDetected = cv2.resize(objectDetected,(80, 80), interpolation = cv2.INTER_LINEAR)
                                carClass=fetchClass(objectDetected)                                
                                if carClass == ['light']:
                                    print "light"
                                    carType=-1
                                if carClass == ['heavy']:
                                    print "heavy"
                                    carType=1
                                if carClass == ['medium']:
                                    print "medium"
                                    carType=2   
                                if carClass == ['other']:
                                    print "other"
                                    carType=0    
                                if(carType==1):
                                    stateHGV=True
                                    timerNow=time.time()                                    
                                    count_HGV+=1
                                    cv2.imwrite("./Heavy/object%d.jpg"%count,objectDetected)
                                if(carType==-1):
                                    count_LTV+=1
                                    cv2.imwrite("./Light/object%d.jpg"%count,objectDetected)
                                if(carType==2):
                                    count_MTV+=1
                                    cv2.imwrite("./Medium/object%d.jpg"%count,objectDetected)                                        
                            elif mse(prevObject,objectDetected)> 3200:
                                count+=1
                                prevObject=objectDetected
                                objectDetected = cv2.resize(objectDetected,(80, 80), interpolation = cv2.INTER_LINEAR)
                                carClass=fetchClass(objectDetected)                                
                                if carClass == ['light']:
                                    print "light"
                                    carType=-1
                                if carClass == ['heavy']:
                                    print "heavy"
                                    carType=1
                                if carClass == ['medium']:
                                    print "medium"
                                    carType=2   
                                if carClass == ['other']:
                                    print "other"
                                    carType==0    
                                #cv2.imwrite("object%d.jpg"%count,objectDetected)
                                #f.write('object: %d,aspect_ratio: %d,solidity: %f,area: %d,extent: %f,circle_dia: %d\n'%(count,aspect_ratio,solidity,area,extent,equi_diameter))
                                if(carType==1):
                                    f.write(' HGV\n')
                                    stateHGV=True
                                    timerNow=time.time()                                    
                                    count_HGV+=1
                                    if not os.path.exists("./Heavy/"):
                                        os.makedirs("./Heavy/")
                                    cv2.imwrite("./Heavy/object%d.jpg"%count,objectDetected)
                                if(carType==-1):
                                    f.write(' LTV\n')
                                    count_LTV+=1
                                    if not os.path.exists("./Light/"):
                                        os.makedirs("./Light/")                                    
                                    cv2.imwrite("./Light/object%d.jpg"%count,objectDetected)
                                if(carType==2):
                                    f.write(' MTV\n')
                                    count_MTV+=1
                                    if not os.path.exists("./Medium/"):
                                        os.makedirs("./Medium/")
                                    cv2.imwrite("./Medium/object%d.jpg"%count,objectDetected)
                            cv2.imshow("object",objectDetected)                    
                            cv2.imshow('MaskApplied',maskApplied)
                            image[tl[1]:br[1], tl[0]:br[0]]=roi
                    
        cv2.imshow("frame", image)
        key = cv2.waitKey(1) & 0xFF            
        if key == ord("i") and len(roiPts) < 4:
    			# indicate that we are in input mode and clone the
    			# frame
           inputMode = True
           cv2.putText(image,'INPUT MODE !',(100,50),0,1,(0,0,255),3)
           while len(roiPts) < 4:
                 
    			cv2.imshow("frame", image)
                   
    			cv2.waitKey(0)
    
    			# determine the top-left and bottom-right points
    			roiPts = np.array(roiPts)
    			s = roiPts.sum(axis = 1)
    			tl = roiPts[np.argmin(s)]
    			br = roiPts[np.argmax(s)]
    
    			# grab the ROI for the bounding box and convert it
    			# to the HSV color space
    			roiBox = (tl[0], tl[1], br[0], br[1])
    
            			# keep looping until 4 reference ROI points have
            			# been selected; press any key to exit ROI selction
            			# mode once 4 points have been selected
       
        if cv2.waitKey(10)== 27:
                break    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    