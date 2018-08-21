import cv2
import os
import pickle
import platform
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


class Vehicle_Detector():
	def __init__(self, model_dict, img_shape):
		self.svc = model_dict['svc']
		self.X_scalar = model_dict['scalar']
		self.orient = model_dict['orient']
		self.pix_per_cell = model_dict['pix_per_cel']
		self.cell_per_block = model_dict['cell_per_block']
		self.spatial_size = model_dict['spatial_size']
		self.hist_bins = model_dict['hist_bins']
		self.color_space = model_dict['color_space']
		self.hog_channel = model_dict['hog_channel']
		self.spatial_feat = model_dict['spatial_feat']
		self.hist_feat = model_dict['hist_feat']
		self.hog_feat = model_dict['hog_feat']
		self.img_shape = img_shape
		self.y_start = img_shape[0] // 2
		self.y_stop = img_shape[0]
		self.scale_list = (1, 1.5, 2, 2.5)
		self.search_range_dict = {1: (0.15, 0.30), 1.5: (0.16, 0.36), 2: (0.15, 0.5), 2.5: (0.2, 0.7)}
		self.heatmap = np.zeros((img_shape[:2]))
		self.rects_queue_length = 10
		self.rects_queue = []  # queue containing rects from each consecutive frame
		self.heatmap_thresh = 30


	def feed(self, img):
		cur_rects = self.get_cur_rects(img)
		self.update_heatmap(cur_rects)
		# self.get_cur_heat_map(cur_rects)
		vehicle_bounding_boxes = self.get_vehicle_bounding_boxes()
		# self.draw_labeled_bboxes(img, vehicle_bounding_boxes)

		return vehicle_bounding_boxes


	def get_cur_rects(self, img_org):
		img = img_org[self.y_start:self.y_stop, :, :]
		draw_img = np.copy(img_org)
		imshape = img.shape
		rects = []
		colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
		color_ind = 0
		for scale in self.scale_list:
			if scale != 1:
				img = cv2.resize(img, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
			if self.color_space != 'BGR':
				if self.color_space == 'HSV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				elif self.color_space == 'LUV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
				elif self.color_space == 'HLS':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
				elif self.color_space == 'YUV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
				elif self.color_space == 'YCrCb':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
				elif self.color_space == 'RGB':
					feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				else:
					feature_image = np.copy(img)
			else:
				feature_image = np.copy(img)

			ch1 = feature_image[:, :, 0]
			ch2 = feature_image[:, :, 1]
			ch3 = feature_image[:, :, 2]

			# compute steps
			nx_blocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
			ny_blocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
			nfeature_per_block = self.orient * self.cell_per_block ** 2
			window = 64
			nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
			cells_per_step = 2
			nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
			ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step + 1
			if self.hog_channel == 'ALL':
				hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block,
											 feature_vec=False)
				hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block,
											 feature_vec=False)
				hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block,
											 feature_vec=False)
			else:
				hog = self.get_hog_features(feature_image[:, :, self.hog_channel], self.orient, self.pix_per_cell,
											self.cell_per_block, feature_vec=False)
			# set search range
			y_search_start = int(self.search_range_dict[scale][0] * ny_steps)
			y_search_end = int(self.search_range_dict[scale][1] * ny_steps)

			for xb in range(nx_steps):
				for yb in range(y_search_start, y_search_end):
					ypos = yb * self.cell_per_block
					xpos = xb * self.cell_per_block
					img_features = []

					xleft = xpos * self.pix_per_cell
					ytop = ypos * self.pix_per_cell

					subimg = cv2.resize(feature_image[ytop:ytop + window, xleft:xleft + window], (64, 64))

					# get color features
					if self.spatial_feat is True:
						spatial_features = self.bin_spatial(subimg, size=self.spatial_size)
						img_features.append(spatial_features)
					if self.hist_feat is True:
						hist_features = self.color_hist(subimg, nbins=self.hist_bins)
						img_features.append(hist_features)
					# get hog features
					if self.hog_feat is True:
						if self.hog_channel == 'ALL':
							hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
							hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
							hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
							hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
						else:
							hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
						img_features.append(hog_features)
					img_features = np.concatenate(img_features)

					test_features = self.X_scalar.transform(img_features.reshape(1, -1))
					test_prediction = self.svc.predict(test_features)

					if test_prediction == 1:
						xbox_left = np.int(xleft * scale)
						ytop_draw = np.int(ytop * scale) + self.y_start
						win_draw = np.int(window * scale)
						rects.append(((xbox_left, ytop_draw), (xbox_left + win_draw, ytop_draw + win_draw)))
						cv2.rectangle(draw_img, (xbox_left, ytop_draw), (xbox_left + win_draw, ytop_draw + win_draw),
									  color=colors[color_ind])
			color_ind += 1

		# cv2.imshow('draw_img', draw_img)
		# cv2.waitKey(0)

		return rects

	def update_heatmap(self, cur_rect):
		self.rects_queue.append(cur_rect)
		for rect in cur_rect:
			self.heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1
		if len(self.rects_queue) == self.rects_queue_length + 1:
			remove_rect = self.rects_queue[0]
			self.rects_queue.pop(0)
			for rect in remove_rect:
				self.heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] -= 1
			# plt.figure(figsize=(10, 10))
			# plt.imshow(self.heatmap, cmap='hot')
			# plt.show()

	def get_vehicle_bounding_boxes(self):
		if len(self.rects_queue) < self.rects_queue_length:
			return []

		valid_heatmap = self.apply_threshold()
		# plt.figure(figsize=(10, 10))
		# plt.imshow(valid_heatmap, cmap='hot')
		# plt.show()
		labels = label(valid_heatmap)
		# print(labels[1], 'cars found')
		# plt.imshow(labels[0], cmap='gray')
		# plt.show()
		rects = []
		for car_number in range(1, labels[1] + 1):
			nonzero = (labels[0] == car_number).nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			rect = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
			rects.append(rect)

		return rects

	def draw_labeled_bboxes(self, img, rects):
		for rect in rects:
			# Define a bounding box based on min/max x and y
			bbox = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]))
			# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 3)

		cv2.imshow('result', img)
		cv2.waitKey(0)

	def apply_threshold(self):
		# Zero out pixels below the threshold
		ret_heatmap = np.copy(self.heatmap)
		ret_heatmap[self.heatmap <= self.heatmap_thresh] = 0

		return ret_heatmap

	def get_cur_heat_map(self, rects):
		heatmap = np.zeros(self.img_shape[:2])
		for rect in rects:
			heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1

		plt.figure(figsize=(10, 10))
		plt.imshow(heatmap, cmap='hot')
		plt.show()
		# cv2.waitKey(0)

	# Define a function to return HOG features and visualization
	def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
						 vis=False, feature_vec=True):
		# Call with two outputs if vis==True
		if vis == True:
			features, hog_image = hog(img, orientations=orient,
									  pixels_per_cell=(pix_per_cell, pix_per_cell),
									  block_norm='L2-Hys',
									  cells_per_block=(cell_per_block, cell_per_block),
									  transform_sqrt=True,
									  visualise=vis, feature_vector=feature_vec)
			return features, hog_image
		# Otherwise call with one output
		else:
			features = hog(img, orientations=orient,
						   pixels_per_cell=(pix_per_cell, pix_per_cell),
						   cells_per_block=(cell_per_block, cell_per_block),
						   block_norm='L2-Hys',
						   transform_sqrt=True,
						   visualise=vis, feature_vector=feature_vec)
			return features

	def bin_spatial(self, img, size=(32, 32)):
		# Use cv2.resize().ravel() to create the feature vector
		features = cv2.resize(img, size).ravel()
		# Return the feature vector
		return features

	def color_hist(self, img, nbins=32, bins_range=(0, 256)):
		# Compute the histogram of the color channels separately
		channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
		channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
		channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return hist_features


if __name__ == '__main__':
	file_name = 'model_dict'
	if os.path.exists(file_name):
		file_object = open(file_name, 'rb')
		model_dict = pickle.load(file_object)
	else:
		raise FileNotFoundError('model_dict not found!')

	if platform.system() == 'Windows':
		test_files = ['.\\test_images\\full_1000960.0.png', '.\\test_images\\full_1000961.0.png',
					  '.\\test_images\\full_1000962.0.png', '.\\test_images\\full_1000963.0.png',
					  '.\\test_images\\full_1000964.0.png', '.\\test_images\\full_1000965.0.png',
					  '.\\test_images\\full_1000966.0.png', '.\\test_images\\full_1000967.0.png',
					  '.\\test_images\\full_1000968.0.png', '.\\test_images\\full_1000969.0.png',
					  '.\\test_images\\full_1000970.0.png', '.\\test_images\\full_1000971.0.png']
	else:
		test_files = ['./test_images/full_1000960.0.png', './test_images/full_1000961.0.png',
					  './test_images/full_1000962.0.png', './test_images/full_1000963.0.png',
					  './test_images/full_1000964.0.png', './test_images/full_1000965.0.png',
					  './test_images/full_1000966.0.png', './test_images/full_1000967.0.png',
					  './test_images/full_1000968.0.png', './test_images/full_1000969.0.png',
					  './test_images/full_1000970.0.png', './test_images/full_1000971.0.png']

	test_image = cv2.imread('./test_images/full_1000960.0.png')
	img_shape = test_image.shape
	windows = [64]
	vehicle_detector = Vehicle_Detector(model_dict, img_shape)
	for test_file in test_files:
		test_image = cv2.imread(test_file)
		rects = vehicle_detector.feed(test_image)
