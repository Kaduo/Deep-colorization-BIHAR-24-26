import skimage
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

input_size = 128
batch_size = 32


class Labelizer:
    def __init__(self):
        self.ab_to_label = {}
        self.label_to_ab = {}
        self.label_counts = {}
        self.last_label = 0

    def ab_pixel_to_label(self, ab_pixel, add=True):
        ab_tuple = (ab_pixel[0], ab_pixel[1])
        if ab_tuple not in self.ab_to_label.keys():
            if add:
                self.ab_to_label[ab_tuple] = self.last_label
                self.label_to_ab[self.last_label] = ab_pixel
                self.label_counts[self.last_label] = 1
                self.last_label += 1
            else:
                return -1
        
        if add:
            self.label_counts[self.ab_to_label[ab_tuple]] += 1
        return self.ab_to_label[ab_tuple]

    def label_to_ab_pixel(self, label):
        return self.label_to_ab[label]

    def ab_image_to_label(self, image):
        return np.apply_along_axis(lambda x : self.ab_pixel_to_label(x), axis=2, arr=image)

    def ab_image_to_label_tensor(self, image):
        return tf.convert_to_tensor(self.ab_image_to_label(image.numpy()))

    def label_image_to_ab(self, image):
        return np.apply_along_axis(lambda x : self.label_to_ab_pixel(x), axis=2, arr=image)
    
    def label_dataset(self, quantized_dataset):

        @tf.py_function(Tout=tf.dtypes.int64)
        def wrap_ab_image_to_label_tensor(x):
            return self.ab_image_to_label_tensor(x)
                
        return quantized_dataset.unbatch().map(
            lambda l, quantized_ab: 
                (l, wrap_ab_image_to_label_tensor(quantized_ab))).batch(batch_size)


def load_base_datasets():
    train, test = keras.utils.image_dataset_from_directory(
                                    "my_celebs",
                                    labels=None,
                                    validation_split=0.2,
                                    subset="both",
                                    seed=1,
                                    image_size=(input_size, input_size), # TODO : remove me
                                )
    return train, test
    

@tf.py_function(Tout=float)
def rgb2lab_tensor(img):
    img_numpy = img.numpy()
    img_numpy /= 255 # TODO : bouger la normalisation ailleurs
    img_lab = skimage.color.rgb2lab(img_numpy)
    return  tf.convert_to_tensor(img_lab)


def rgb2lab_dataset(rgb_dataset):
    return rgb_dataset.unbatch().map(rgb2lab_tensor).batch(batch_size)

def split_l_and_ab_dataset(lab_dataset):
    l_batch_shape = [None, 128, 128]
    ab_batch_shape = [None, 128, 128, 2]

    def split_l_and_ab_batch(lab_batch):
        l_batch = lab_batch[:,:,:,0]
        ab_batch = lab_batch[:,:,:,1:]
        return (tf.ensure_shape(l_batch, l_batch_shape),
                tf.ensure_shape(ab_batch, ab_batch_shape))
    
    return lab_dataset.map(split_l_and_ab_batch)

def quantize_dataset(split_dataset):

    def quantize_ab_batch(ab_batch):
        return (ab_batch//10)*10 + 5
    
    return split_dataset.map(lambda l_batch, ab_batch: (l_batch, quantize_ab_batch(ab_batch)))


def create_model():
    conv_param_dict = {"kernel_size": (3,3),
                       "strides": 2,
                       "padding": "same",
                       "activation":
                       "relu"}
    
    layers = [
        keras.Input(input_size, input_size),
        keras.layers.Reshape((input_size, input_size, 1)),
        keras.layers.Rescaling(scale=1./50, offset=-1),
    ]

    depths = [64, 128, 256, 512, 1024]
    for d in depths:
        layers.append(keras.layers.Conv2D(d, **conv_param_dict))
        layers.append(keras.layers.BatchNormalization())

    for d in depths[:-1:-1]:
        layers.append(keras.layers.Conv2DTranspose(d, **conv_param_dict))
        layers.append(keras.layers.BatchNormalization())

    layers.append(
        keras.layers.Conv2DTranspose(
            2,
            kernel_size=(3,3),
            strides=2,
            padding="same",
            activation="tanh"
        ),
    )

    layers.append(keras.layers.Rescaling(scale=128.))

    return keras.Sequential(layers)


