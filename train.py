import cv2
import numpy as np
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import platform
import pickle
import os
import sys
import time

# Global parameters
COLOR_SPACE = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9  # HOG orientations
PIX_PER_CELL = 8  # HOG pixels per cell
CELL_PER_BLOCK = 2  # HOG cells per block
HOG_CHANNEL = 0  # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (32, 32)  # Spatial binning dimensions
HIST_BINS = 32  # Number of histogram bins
SPATIAL_FEAT = True  # Spatial features on or off
HIST_FEAT = True  # Histogram features on or off
HOG_FEAT = True  # HOG features on or off


class Train():
    def __init__(self, car_imgs_folder, noncar_imgs_folder, color_space='BGR', spatial_size=(32, 32),
                 hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        self.car_paths = []
        self.noncar_paths = []
        self.car_imgs_folder = car_imgs_folder
        self.noncar_imgs_folder = noncar_imgs_folder
        self.svc = LinearSVC()
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.X_scalar = None

    def train(self):
        self.update_imgpaths_lists()
        print('Image paths updated with:\n\tVehicle Images:{}\n\tNon-vehicle Images:{}'.format(len(self.car_paths),
                                                                                               len(self.noncar_paths)))
        print('Start extracting car features...')
        car_features = self.extract_features(self.car_paths)
        print('Finished car feature extraction!')
        print('Start extracting noncar features...')
        noncar_features = self.extract_features(self.noncar_paths)
        print('Finished noncar feature extraction!')

        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        # split data
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

        # normalize data
        self.X_scalar = StandardScaler().fit(X_train)
        X_train = self.X_scalar.transform(X_train)
        X_test = self.X_scalar.transform(X_test)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))


    def get_model_dict(self):
        dict = {}
        dict['svc'] = self.svc
        dict['scalar'] = self.X_scalar
        dict['orient'] = self.orient
        dict['pix_per_cel'] = self.pix_per_cell
        dict['cell_per_block'] = self.cell_per_block
        dict['spatial_size'] = self.spatial_size
        dict['hist_bins'] = self.hist_bins
        dict['color_space'] = self.color_space
        dict['hog_channel'] = self.hog_channel
        dict['spatial_feat'] = self.spatial_feat
        dict['hist_feat'] = self.hist_feat
        dict['hog_feat'] = self.hog_feat

        return dict

    def update_imgpaths_lists(self):
        if self.car_imgs_folder is None or self.noncar_imgs_folder is None:
            raise ValueError('Image foler is None!')

        # udpate car images paths
        for item in os.listdir(self.car_imgs_folder):
            if platform.system() == 'Windows':
                path = self.car_imgs_folder + item + '\\'
            else:
                path = self.car_imgs_folder + item + '/'
            if os.path.isdir(path):
                images = glob.glob(path + '*.png')
                self.car_paths.extend(images)
        # udpate non-car images paths
        for item in os.listdir(self.noncar_imgs_folder):
            if platform.system() == 'Windows':
                path = self.noncar_imgs_folder + item + '\\'
            else:
                path = self.noncar_imgs_folder + item + '/'
            if os.path.isdir(path):
                images = glob.glob(path + '*.png')
                self.noncar_paths.extend(images)

    def extract_features(self, img_paths):
        features = []
        ind = 0
        for file in img_paths:
            ind += 1
            if ind % 500 == 0:
                sys.stdout.write('\r\tFinished: ' + str(ind) + '/' + str(len(img_paths)))
                time.sleep(0.5)
            image = cv2.imread(file)
            img_features = []
            # apply color conversion if other than 'BGR'
            if self.color_space != 'BGR':
                if self.color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif self.color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif self.color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif self.color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                elif self.color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                elif self.color_space == 'RGB':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError('Specified color space not supported!')
            else:
                feature_image = np.copy(image)

            if self.spatial_feat is True:
                spatial_features = self.bin_spatial(feature_image, size=self.spatial_size)
                img_features.append(spatial_features)

            if self.hist_feat is True:
                hist_features = self.color_hist(feature_image, nbins=self.hist_bins)
                img_features.append(hist_features)

            if self.hog_feat is True:
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel], orient=self.orient,
                                                                  pix_per_cell=self.pix_per_cell,
                                                                  cell_per_block=self.cell_per_block,
                                                                  vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], orient=self.orient,
                                                         pix_per_cell=self.pix_per_cell,
                                                         cell_per_block=self.cell_per_block,
                                                         vis=False, feature_vec=True)
                img_features.append(hog_features)

            features.append(np.concatenate(img_features))
        print()

        return features

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

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


if __name__ == '__main__':
    if platform.system() == 'Windows':
        car_imgs_folder = '.\\images_data\\vehicles\\'
        noncar_imgs_folder = '.\\images_data\\non-vehicles\\'
    else:
        car_imgs_folder = './images_data/vehicles/'
        noncar_imgs_folder = './images_data/non-vehicles/'

    file_name = 'model_dict'
    if os.path.exists(file_name):
        file_object = open(file_name, 'rb')
        model_dict = pickle.load(file_object)
    else:
        train = Train(car_imgs_folder, noncar_imgs_folder, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                      hist_bins=HIST_BINS, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK,
                      hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        train.train()
        model_dict = train.get_model_dict()
        # save model
        file_object = open(file_name, 'wb')
        pickle.dump(model_dict, file_object)
        file_object.close()

    svc = model_dict['svc']
