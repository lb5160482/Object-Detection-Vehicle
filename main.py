import cv2
from Vehicle_Detector import Vehicle_Detector
import os
import pickle


cap = cv2.VideoCapture('./test_video.mp4')

file_name = 'model_dict'
if os.path.exists(file_name):
	file_object = open(file_name, 'rb')
	model_dict = pickle.load(file_object)
else:
	raise FileNotFoundError('model_dict not found!')
test_file = './test_images/test1.jpg'
test_image = cv2.imread(test_file)
img_shape = test_image.shape
vehicle_detector = Vehicle_Detector(model_dict, img_shape)
video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
							   (img_shape[1], img_shape[0]))
while cap.isOpened():
	ret, frame = cap.read()
	vehicle_bounding_boxes = vehicle_detector.feed(frame)
	for rect in vehicle_bounding_boxes:
		# Define a bounding box based on min/max x and y
		bbox = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]))
		# Draw the box on the image
		cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 3)
	video_writer.write(frame)
	cv2.imshow('result', frame)
	cv2.waitKey(1)