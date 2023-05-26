import pywt
import numpy as np
import pywt.data
from PIL import Image
import os
from sklearn.decomposition import PCA


np.random.seed(42)

cwd = os.getcwd() # current working directory
dataset_path = cwd + "/data/CK+48"
characteristic_vectors_path = cwd + "/data/characteristic_vectors"
csv_datafile = 'VC.csv'

wavelet_families = [ # experimentacion
    'db2', 'db4', 'db8',
    'sym2', 'sym4', 'sym8',
    'coif2', 'coif4', 'coif6',
    'bior2.2', 'bior4.4', 'bior6.8'
]

# experimentacion
# Realizar reduccion de la dimensiion con varios n
# Hacer las pruebas con 200 dim, 150 dim, 100 dim, etc.

if not os.path.exists(characteristic_vectors_path):
    os.makedirs(characteristic_vectors_path)

def process_dataset(csvs = 1):
    data = []
    # ndim = 0
    flag = True
    dataset_subdir_list = os.listdir(dataset_path) # anger, contempt, disgust, etc...
    for subdir in dataset_subdir_list:
        subdir_path = dataset_path + "/" + subdir
        for image in os.listdir(subdir_path):
            image_path = subdir_path + "/" + image
            original = np.array(Image.open(image_path))
            coeffs = pywt.dwt2(original, 'haar') # Perform 2D DWT using Haar wavelet
            approximation = coeffs[0]
            characteristic_vector = np.ravel(approximation)
            if flag: # to avoid calculating len() each iteration
                max_dim = len(characteristic_vector) # 576 DIM
                flag = False
            data.append(characteristic_vector)
    
    dimensions = np.arange(2, max_dim, round(max_dim/csvs))
    dimensions = np.append(dimensions, max_dim)
    data = np.array(data)

    for dim in dimensions:
        pca = PCA(n_components = dim)
        reduced_data = pca.fit_transform(data)
        n_csv_datafile = str(dim) + "_" + csv_datafile
        matrix_path = characteristic_vectors_path + "/" + n_csv_datafile
        np.savetxt(matrix_path, reduced_data, delimiter=',')
        print('Data stored in', characteristic_vectors_path)
    # return matrix_path