# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import dlib
import faceAlignment as fa
import sys
import select
import glob
import os
import predict as pred
import concate as conc
import time

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

def find_face_landmark(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)

	detections = detector(clahe_image, 1) #Detect the faces in the image

	for k,d in enumerate(detections): #For each detected face
	    shape = predictor(clahe_image, d) #Get coordinates
	    for i in range(1,68): #There are 68 landmark points on each face
	        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
	return frame

def clean_pictures():
	#clean folder
	files = glob.glob('pictures/*')
	for f in files:
	    os.remove(f)
	files = glob.glob('result_lip/*')
	for f in files:
	    os.remove(f)

def align_face_and_resize_frames():
	# segement picture
	img_index=-1
	# read in reverse becasue the sequence of pictures is generated in the descending order
	for f in ['30.jpg','28.jpg','26.jpg','24.jpg','22.jpg','20.jpg','18.jpg','16.jpg','14.jpg','12.jpg','10.jpg','8.jpg','6.jpg','4.jpg','2.jpg']:
	    img_index +=1
	    im = cv2.imread("pictures/"+f, cv2.IMREAD_COLOR)
	    print("processing "+f)
	    lip_frame = fa.gen_align_lip_from_video(im)
	    lip_frame = cv2.resize(lip_frame, (35, 35))#resize
	    cv2.imwrite('result_lip/' + str(img_index) + '.jpg', lip_frame)

# construct the argument parse and parse the arguments
# n is the max iteration number the program waits for "press Enter"
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=1000,
	help="# of frames to loop over for FPS test")
args = vars(ap.parse_args())

while True:
	clean_pictures()
	print("[INFO] sampling THREADED frames from webcam...")
	vs = WebcamVideoStream(src=0).start()
	fps = FPS().start()
	record_index=-1

	# loop over some frames...this time using the threaded stream
	while fps._numFrames < args["num_frames"]:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = cv2.resize(frame, (512, 256))
		triggered=heardEnter()

		print("frames: "+str(fps._numFrames)+" heardEnter: "+str(triggered)+ " record_index: "+str(record_index))

		if triggered==True:
			record_index=30
			print("triggered")
		if record_index>0:
			if record_index%2==0:
				cv2.imwrite("pictures/"+str(record_index)+".jpg", frame)
			else:
				cv2.imwrite("dump.jpg", frame)
			record_index=record_index-1
		elif record_index==0:
			break;
		else:	
			cv2.imwrite("dump.jpg", frame)
		key= 0xFF & cv2.waitKey(35)
	    # update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	vs.stop()

	#waiting too long
	if fps._numFrames == args["num_frames"]:
		break
	align_face_and_resize_frames() # process images under pictures and store under folder result_lip
	conc.concate_images() # concate 15 images under folder result_lip

	#call classifier
	if_user_say_bye = pred.predict_by_model("result_lip/concate-output.jpg")
	if if_user_say_bye==0:
		break
	#sleep for 1 s
	time.sleep(1)


