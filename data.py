import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import imutils

class Data(object):
    def __init__(self, data_dir, height, width, batch_size):
        if data_dir[-1] != "/":
            data_dir = data_dir + "/"
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.batch_size = batch_size

    def get_rot_data_iterator(self, images, labels, batch_size):
        # TODO: Initialize your iterator with these images and labels
        # You should be prefetching, shuffling, and batching your data. See the tf.data documentation is you are confused on this part.
		# There are multiple ways to implement iterators. Feel free to do it in whichever way makes the most sense.
        dataset = tf.data.Dataset.from_tensor_splices((images, labels))
        dataset.shuffle.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def get_training_data(self):
        print("[INFO] Getting Training Data")
        #TODO: This function should return the training images and labels. You can use the helper functions for this.
        images = []
        labels = []
        for i in range(1,6):
            images.append(self._get_next_batch_from_file(i)[b'data'])
            labels.append(self._get_next_batch_from_file(i)[b'labels'])
        return self.convert_images(np.concatenate(images)), np.concatenate(labels)

        #data = {}
        #for i in range(1, 6):
        #    d = self._get_next_batch_from_file(i)
        #    data.update(d)

        #images, labels = data[b'data'], data[b'labels']
        #images = self.convert_images(images)
        #images, labels = self.preprocess(images)
        #return images, labels

    def convert_images(self, raw_images):
        #This function normalizes the input images and converts them to the appropriate shape: batch_size x height x width x channels
        images = raw_images / 255.0
        images = raw_images.reshape([-1, 3, self.height, self.width])
        images = images.transpose([0, 2, 3, 1])
        return images

    def _get_next_batch_from_file(self, batch_number):
        data = self._unpickle_data(self.data_dir + self._get_batch_name(batch_number))
        return data

    def _get_batch_name(self, number):
        return "data_batch_{0}".format(number)

    def _unpickle_data(self, filepath):
        with open(filepath, 'rb') as data:
            dict = pickle.load(data, encoding='bytes')
        return dict

    def get_test_data(self):
        # TODO: Write a function to get the test set from disk.
        d = self._unpickle_data(self.data_dir + 'test_batch')
        images = self.convert_images(images)
        return images, labels

    def preprocess(self, images):
        #TODO: Rotate your images and save the labels for each rotation. Search google to figure out how to rotate
        #The output should be a tuple of your images and labels

        images_rotated = []
        labels_rotated = []
        for num_r in [0, 1, 2, 3]:
            for i in range(len(images)):
                r_im = np.rot90(images[i], k=num_r).reshape(1, 32, 32, 3)
                images_rotated.append(r_im)
                labels_rotated.append(num_r * 90)
        return np.concatenate(images_rotated), np.array(labels_rotated)

        #for image in im:
        #    for i in range(4):
        #        images.append(np.rot90(image, k=i))

        #for v in lab:
        #    for i in range(4):
        #        labels.append(v)
        #return np.array(images), np.array(labels)

    @staticmethod
    def print_image_to_screen(data):
        """
        Used for debugging purposes. You can use this to see if your image was actually rotated.
        """
        img = Image.fromarray(data, 'RGB')
        img.show()

    @staticmethod
    def get_image(image_path):
        #TODO: Load a single image for inference given the path to the data
        #This is not required but can be used in your predict method in rotnet.py
        ...
        return

if __name__ == "__main__":
    #You can use this for testing
    DATA_DIR = "./data/cifar-10-batches-py/"
    data_obj = Data(DATA_DIR, 32, 32, 5000)
    x, y = data_obj.get_training_data()
    xr, yr = data_obj.preprocess(x)
    data_obj.print_image_to_screen(xr[50000]) #50000 gives rotation 1, 0 gives initial image
