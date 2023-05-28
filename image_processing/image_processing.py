import pywt
import numpy as np
import pywt.data
import os
import cv2
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

cwd = os.getcwd()
dataset_path = cwd + "/data/CK+48"
characteristic_vectors_path = cwd + "/data/characteristic_vectors"
wavelet_csv_datafile = 'wavelet_feature_vectors_of_'

if not os.path.exists(characteristic_vectors_path):
    os.makedirs(characteristic_vectors_path)

def process_dataset(csvs = 1):
    wavelet_data = []
    real_labels = []

    dataset_subdir_list = os.listdir(dataset_path) # anger, contempt, disgust, etc...
    for subdir in dataset_subdir_list:
        subdir_path = dataset_path + "/" + subdir
        
        for image in sorted(os.listdir(subdir_path)):
            image_path = subdir_path + "/" + image

            # Load Image
            original = cv2.imread(image_path)

            # convert to grayscale
            gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            # WAVELETS - haar - Wavelet transform
            coeffs = pywt.dwt2(gray_image, 'haar')

            # Extract wavelet coefficients
            cA, (cH, cV, cD) = coeffs

            # Normalize coefficients
            cA_normalized = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cH_normalized = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cV_normalized = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cD_normalized = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Extract LBP features from each coefficient
            lbp_cA = local_binary_pattern(cA_normalized, 8, 1, method='uniform')
            lbp_cH = local_binary_pattern(cH_normalized, 8, 1, method='uniform')
            lbp_cV = local_binary_pattern(cV_normalized, 8, 1, method='uniform')
            lbp_cD = local_binary_pattern(cD_normalized, 8, 1, method='uniform')

            # Concatenate the LBP features
            lbp_features = np.concatenate((lbp_cA.ravel(), lbp_cH.ravel(), lbp_cV.ravel(), lbp_cD.ravel()))

            wavelet_data.append(lbp_features)

            real_labels.append(subdir)

    wavelet_data = np.array(wavelet_data)

    # normalize data
    scaler          = MinMaxScaler()
    wavelet_data    = scaler.fit_transform(wavelet_data)

    # reduce dimension using PCA
    dims = np.arange(2, len(wavelet_data), int(np.floor( len(wavelet_data)/csvs)))
    for dim in dims:
        # using pca to reduce dimension
        pca = PCA(n_components=dim)
        pca.fit(wavelet_data)
        reduced_data = pca.transform(wavelet_data)

        # save data
        data_path = characteristic_vectors_path + "/" + wavelet_csv_datafile + "_" + str(dim) + ".csv"
        np.savetxt(data_path, reduced_data, delimiter=',')

    # save data - without dimension reduction - 981 dimensions
    data_path = characteristic_vectors_path + "/" + wavelet_csv_datafile + "_" +  str(len(wavelet_data)) + ".csv"
    np.savetxt(data_path, wavelet_data, delimiter=',')

    return real_labels


def get_real_labels():
    real_labels = []
    emotions = {'anger':0, 'contempt':1, 'disgust':2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6}


    dataset_subdir_list = os.listdir(dataset_path) # anger, contempt, disgust, etc...
    for subdir in dataset_subdir_list:
        subdir_path = dataset_path + "/" + subdir
        
        for _ in sorted(os.listdir(subdir_path)):
            real_labels.append(emotions[subdir])

    return real_labels


