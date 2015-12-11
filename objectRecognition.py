# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
#from RPIO import PWM
import time
import cv2
import numpy as np
 
MATCH_COUNT = 5


#SURF Feature Initialization
surf = cv2.xfeatures2d.SURF_create(400)

img1 = cv2.imread('testchip2')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kp1, des1 = surf.detectAndCompute(gray1, None)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
height = 320
width = 240
center_x = height/2
center_y = width/2
camera.resolution = (height,width)
camera.framerate = 90
camera.brightness = 54
camera.vflip = True

rawCapture = PiRGBArray(camera, size=(height, width))
 
# allow the camera to warmup
time.sleep(0.1)

	
def eliminate_Noises(image):
	#Construct threshhold image for coloritem, then perform
	#a series of dilation and erosions to remove any small 
	#blobs left in the threshold image.
    
    ret, thresh = cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #(thresh, 127,255, 0)  
        
    
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    return thresh
# capture frames from the camera
def DectectandComputeKP_DES(kp1,des1,img1,img2):

    
		#Finding keypoints and descriptors of image from video
        kp2,des2 = surf.detectAndCompute(img2, None)
		
		#Detect keypoints on image and computer to see how many descriptors found from image.
		#FLANN Parameters
        index_params = dict(algorithm = 0, trees = 5)
		
        #Pass empty dictionary
        search_params = dict(checks=50)   
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
		#find matches of both image.
        matches = flann.knnMatch(des1,des2,k=2)  
        	
        print "Ori_Image: KP= "+ str(len(kp1)) + ", Des= " + str(len(des1))   \
              + " | Check Image: KP= "+str(len(kp2)) + ", Des= " + str(len(des2))
       
           
    #Apply ratio test

        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
			    
        
        if len(good) > MATCH_COUNT:
            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)        

            M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w,d = img1.shape
            pts = np.float32([[0,0,0],[0,h-1,d],[w-1,h-1,d-1],[w-1,0,d]]).reshape(-1,1,4)
            dst = cv2.perspectiveTransform(pts,M)
            
			#Drawing the lines along the image size in order to mark target found
            img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(img2,'Object Found, BINGO!', (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(73,222,27),3) 
        else:
            cv2.putText(img2,'Object NOT Found!', (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3) 
            print"Not Enough Matches are found - %d/%d" % (len(good), MATCH_COUNT)
            matchesMask= None
    return matchesMask

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        img2 = frame.array
		
		#Convert BGR to HSV Color space	
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		
		#Eliminate blobs and other noises in the image
		img1 = eliminate_Noises(gray1)
		img2 = eliminate_Noises(gray2)
		
		#Detect and computer the keypoint and descriptors of image.
		#Finding best matches between image and frame captured from video
 	    DectectandComputeKP_DES(kp1,des1,img1,img2):
        
            
        draw_matches = dict(matchColor = (0,255,0), singlePointColor=None,matchesMask =matchesMask,flags=2)			            

		#draw first 10 matches
        result = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_matches)
        
        #cv2.imshow("GrayScale Detection",fgmask)
        cv2.imshow("Final Image",result)     
       
        key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
        rawCapture.seek(0)
        rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
        	break
