import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def compute_gradcam(model, image, labels, layer_name='224_block5_pool', W=224, H=224):

    """
    Function to compute GradCAM at a specific layer of a trained model.

    Inputs
    model: Trained model object
    image: Array-like image to evaluate
    labels: List of labels
    layer_name: Name of the layer to evaluate
    W: With of the desired image
    H: Height of the desired image    
    """

    predictions = model.predict(image)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(image[0,:,:,0], cmap='gray')

    j = 1
    for i in range(len(labels)):  
        print(f"Generating gradcam for class {labels[i]}")

        y_c = model.output[0, i]
        conv_output = model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]

        gradient_function = K.function([model.input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))
        gradcam = np.dot(output, weights)

        # Process CAM
        gradcam = cv2.resize(gradcam, (W, H), cv2.INTER_LINEAR)
        gradcam = np.maximum(gradcam, 0)
        if gradcam.max()!=0:
            print(gradcam.max())
            gradcam = gradcam / gradcam.max()

        plt.subplot(151 + j)
        plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
        plt.axis('off')
        plt.imshow(image[0,:,:,0],cmap='gray')
        plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
        j += 1
