import numpy as np
import cv2

#read from folder result_lip and write to same folder with name concate-output.jpg
def concate_images(): 
	concate_seq= np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,10,11,12,13,14,14])
	#read image
	path="result_lip/"
	im_1 = cv2.imread(path+"0.jpg", cv2.IMREAD_COLOR)
	im_2 = cv2.imread(path+"0.jpg", cv2.IMREAD_COLOR)
	im_3 = cv2.imread(path+"1.jpg", cv2.IMREAD_COLOR)
	im_4 = cv2.imread(path+"1.jpg", cv2.IMREAD_COLOR)
	im_5 = cv2.imread(path+"2.jpg", cv2.IMREAD_COLOR)
	layer1 = np.concatenate((im_1, im_2, im_3, im_4, im_5), axis=1)

	im_6 = cv2.imread(path+"2.jpg", cv2.IMREAD_COLOR)
	im_7 = cv2.imread(path+"3.jpg", cv2.IMREAD_COLOR)
	im_8 = cv2.imread(path+"3.jpg", cv2.IMREAD_COLOR)
	im_9 = cv2.imread(path+"4.jpg", cv2.IMREAD_COLOR)
	im_10 = cv2.imread(path+"4.jpg", cv2.IMREAD_COLOR)
	layer2 = np.concatenate((im_6, im_7, im_8, im_9, im_10), axis=1)

	im_11 = cv2.imread(path+"5.jpg", cv2.IMREAD_COLOR)
	im_12 = cv2.imread(path+"5.jpg", cv2.IMREAD_COLOR)
	im_13 = cv2.imread(path+"6.jpg", cv2.IMREAD_COLOR)
	im_14 = cv2.imread(path+"6.jpg", cv2.IMREAD_COLOR)
	im_15 = cv2.imread(path+"7.jpg", cv2.IMREAD_COLOR)
	layer3 = np.concatenate((im_11, im_12, im_13, im_14, im_15), axis=1)

	im_16 = cv2.imread(path+"7.jpg", cv2.IMREAD_COLOR)
	im_17 = cv2.imread(path+"8.jpg", cv2.IMREAD_COLOR)
	im_18 = cv2.imread(path+"8.jpg", cv2.IMREAD_COLOR)
	im_19 = cv2.imread(path+"9.jpg", cv2.IMREAD_COLOR)
	im_20 = cv2.imread(path+"10.jpg", cv2.IMREAD_COLOR)
	layer4 = np.concatenate((im_16, im_17, im_18, im_19, im_20), axis=1)

	im_21 = cv2.imread(path+"11.jpg", cv2.IMREAD_COLOR)
	im_22 = cv2.imread(path+"12.jpg", cv2.IMREAD_COLOR)
	im_23 = cv2.imread(path+"13.jpg", cv2.IMREAD_COLOR)
	im_24 = cv2.imread(path+"14.jpg", cv2.IMREAD_COLOR)
	im_25 = cv2.imread(path+"14.jpg", cv2.IMREAD_COLOR)
	layer5 = np.concatenate((im_21, im_22, im_23, im_24, im_25), axis=1)

	output = np.concatenate((layer1, layer2, layer3, layer4, layer5), axis=0)
	cv2.imwrite(path+"concate-output.jpg", output)
	print("[INFO] concate done")



#concate_images()