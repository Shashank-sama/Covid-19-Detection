import numpy as np

def samplewise_preprocessing(images):
    
    """
    Function to apply Z-score Normalization to a set of images. Resulting images will be mean 0 and standard deviation 1.

    Inputs
    images: Array-like set of images

    Outputs
    Processed images, Labels, Mean and Standard Deviation over the whole set
    """
    
    processed_images = []
    means = []
    stds = []
    for i in range(images.shape[0]):
        mean = np.mean(images[i])
        std = np.std(images[i])
        if std!=0 and mean !=0:
            means.append(mean)
            stds.append(std)
            processed_images.append((images[i]-mean)/std)
    return np.array(processed_images), np.mean(means), np.mean(stds)



def featurewise_preprocessing(images, mean, std):

    """
    Function to apply Z-score Normalization to a set of images using a value for the mean and standard deviation.

    Inputs
    images: Array-like set of images
    mean: Value for the mean
    std: Value for the standard deviation

    Outputs
    Processed images
    """

    processed_images = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        processed_images[i] = (images[i]-mean)/std
    return processed_images



def min_max_preprocessing(images):

    """
    Function to apply min-max normalization to a set of images. Resulting images will be in range [0, 1].

    Inputs
    images: Array-like set of images

    Outputs
    Processed images
    """

    processed_images = []
    for i in range(len(images)):
        try:
          maxi=np.max(images[i])
          mini=np.min(images[i])
          if (maxi-mini)!=0:
            processed_images.append((images[i]-mini)/(maxi-mini))
        except:
          continue
    return np.array(processed_images)



def preprocess_images(train_images, dev_images, test_images, min_max=True, z_score=True):
    
    """
    Function to apply min-max and/or Z-score normalization to a set of images

    Inputs
    train_images: Array of training images
    dev_images: Array of validation images
    test_images: Array of test images
    min_max: Apply or not min-max normalization (bool)
    z_score Apply or not z-score normalization (bool)
    """

    if min_max:
        X_train = min_max_preprocessing(train_images)
        X_dev   = min_max_preprocessing(dev_images)
        X_test  = min_max_preprocessing(test_images)
    if z_score:
        X_train, mean, std = samplewise_preprocessing(X_train)
        X_dev  = featurewise_preprocessing(X_dev, mean, std)
        X_test = featurewise_preprocessing(X_test, mean, std)
    return X_train, X_dev, X_test