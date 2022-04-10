import os
import cv2
from tqdm import tqdm
from sklearn.utils import class_weight
import numpy as np

def checkDuplicates(trainDF, devDF, testDF, id_column):
    
    """
    Function to check for duplicate values between Dataframes in a specific column.

    Inputs
    trainDF: DataFrame containing the information of training data
    devDF: DataFrame containing the information of validation data
    testDF: DataFrame containing the information of test data
    id_column: Name of the column to evaluate
    """
    
    patientsTrain = set(trainDF[id_column])
    patientsDev = set(devDF[id_column])
    patientsTest = set(testDF[id_column])
    
    ids = list(patientsTrain.intersection(patientsDev))
    print('# de pacientes de train presentes en dev:', len(ids))
    
    ids_ = list(patientsTrain.intersection(patientsTest))
    print('# de pacientes de train presentes en test:', len(ids_))
    
    ids.extend(ids_)
    ids_dev = list(patientsDev.intersection(patientsTest))
    print('# de pacientes de dev presentes en test:', len(ids_dev))

    return ids, ids_dev



def saveNPY(DF, destination, name, src_dir, src_column, W=224, H=224):
    
    """
    Function to combine images in a directory structure into a single NPY file for faster import.

    Inputs
    DF: DataFrame containing filenames for the images
    destination: Path to destination folder
    name: Filename for the resulting .npy file
    src_dir: Path to the source directory of images
    src_column: Name of the column of the DataFrame that contains the filenames
    W: Desired width of the images
    H: Desired heigth of the images
    """

    images = []
    print('reading images...')
    #Read images in png format
    for i in tqdm(DF[src_column]):
        src_file = os.path.join(src_dir, i)
        img = cv2.imread(src_file, -1)
        resized = cv2.resize(img, (W, H))
        if resized.shape==(W, H, 4):
            images.append(resized[:,:,0])
        else:
            images.append(resized)
    NPY = np.array(images)
    images_filename = destination+name+'.npy'
    np.save(images_filename, NPY)
    print('done!')



def compute_class_freqs(labels):
    
    """
    Function to compute the class frequencies for a binary task.

    Inputs
    labels: Ground truth labels
    """
    
    N = labels.shape[0]
    positive_frequencies = np.sum(labels,axis=0)/N
    negative_frequencies = 1-positive_frequencies
    return positive_frequencies, negative_frequencies



def convert_n_CH(train_images, dev_images, test_images, mode='gray'):
    
    """
    Function to adjust the number of channels in grayscale images to feed a model.
    'gray' mode adds a dimension to the array representing 1 channel and
    'rgb' mode replicates the grayscale image data into 3 channels 

    Inputs
    train_images: Array of images for training.
    dev_images: Array of images for validation.
    test_images: Array of images for test.
    mode: 'gray' or 'rgb'
    """

    if mode=='gray':
        converted_train = np.expand_dims(train_images,axis=3)
        converted_dev   = np.expand_dims(dev_images,axis=3)
        converted_test  = np.expand_dims(test_images,axis=3)
    elif mode=='rgb':
        converted_train = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in tqdm(train_images)])
        converted_dev   = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in tqdm(dev_images)])
        converted_test  = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in tqdm(test_images)])
    return converted_train, converted_dev, converted_test



def get_weight(y):

    """
    Function to compute class weights.

    Inputs
    y: List of labels
    """

    class_weight_current =  class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_current=dict(enumerate(class_weight_current.flatten(), 0))
    return class_weight_current