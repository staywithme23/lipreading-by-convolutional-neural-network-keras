import numpy as np
import cv2
import os.path
import time

def display_text(textString):
	# Create a black image
	img = np.zeros((100,512,3), np.uint8)

	# Write some Text
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, textString,(10,50), font, 1,(255,255,255),2)

	#Display the image
	cv2.namedWindow('text', cv2.WINDOW_NORMAL)
	cv2.imshow("text",img)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	while True:
		if os.path.exists("result_lip/concate-output.jpg") and os.path.exists("result_lip/text.txt"):
			with open("result_lip/text.txt") as f:
				content = f.readlines()
				print(content[0])
				display_text(content[0])
				if(content[0]=="Good bye"):
					break
		else:
			display_text("Processing")
	time.sleep(1)		