# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
#from RPIO import PWM
import time
import cv2
import numpy as np
 
#Servos Setup
#GPIO.setmode(GPIO.BOARD)
#tilt_ser = 11
#GPIO.setup(13,GPIO.OUT)
#GPIO.setup(tilt_ser,GPIO.OUT)
#p = GPIO.PWM(tilt_ser,100)
#p.start(5)

minSquare = 5000
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
height = 640
width = 480
center_x = height/2
center_y = width/2
camera.resolution = (height,width)
camera.framerate = 90
camera.brightness = 54
#camera.hflip = True
camera.vflip = True

rawCapture = PiRGBArray(camera, size=(height, width))
 
# allow the camera to warmup
time.sleep(0.1)

def auto_canny(image,sigma=0.50):
	#compute the median of the single channel pixel intesities
	v = np.median(image)
	
	#apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower,upper)
	
	return edged
	
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

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        image = frame.array	
		
        blurr = cv2.GaussianBlur(image, (5,5), 0)
        

		#Convert BGR to HSV Color space
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        
        #Blur the image to reduce high frequency noise
		#Better focus on structural objects.
        blurred = cv2.GaussianBlur(hsv, (5,5), 0)
        
        
        
        thresh = auto_canny(blurred)
        #thresh = cv2.inRange(hsv,np.array((0, 200, 200)), np.array((20, 255, 255)))
        thresh = eliminate_Noises(thresh)
		
		#blue Color 76 31 4, 210 90 70 , dtype="uint8"
        lower = np.array([20,100,100],dtype="uint8")
        upper = np.array([50,255,255],dtype="uint8")
        thresh = cv2.inRange(hsv, lower, upper)
				
        # find contours in the threshold image
        image, contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		#center = None		
        
        # finding contour with maximum area and store it as best_cnt
        max_area = 0
        best_cnt = 1
		
        for cnt in contours:
            area = cv2.contourArea(cnt)
            cv2.drawContours(blurr,cnt, 0 , (0,0,255), 3)
                
            if area > max_area:
                        max_area = area
                        best_cnt = cnt
		
        # finding centroids of best_cnt and draw a circle there
        M = cv2.moments(best_cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        cv2.line(blurr,(cx,cy-15),(cx,cy+15),(0,255,255), 3)
        cv2.line(blurr,(cx+-15,cy),(cx+15,cy),(0,255,255), 3)
        cv2.putText(blurr, str(cx) + "," + str(cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(73,222,27),3) 
        
        dx = cx - (height/2)
        dy = cy - (width/2)
                 
           
        cv2.imshow("Final Image",blurr)
        cv2.imshow('thresh',blurred)
       
        key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
        rawCapture.seek(0)
        rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
        	break
