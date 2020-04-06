import yaml
import os
import sys
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from resnet import ResNet
from data import Data

class RotNet(object):
    def __init__(self, sess, args):
        #TODO: Look through this function to see which attributes have already been initalized for you.
        print("[INFO] Reading configuration file")
        tf.compat.v1.disable_v2_behavior()
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number
        self.model = ResNet()

        self._populate_model_hyperparameters()
        self.data_obj = Data(self.data_dir, self.height, self.width, self.batch_size)
        self.build_base_graph()

        if args.train:
            #If we are training, then we want to run the optimizer
            self.build_train_graph()

        #List the compute available on the device that this script is being run on.
        print(device_lib.list_local_devices())

        #This collects the add_summary operations that you defined in the graph. You should be saving your metrics to self.summary
        self.summary = tf.compat.v1.summary.merge_all()

    def _populate_model_hyperparameters(self):
        """
        This is a helper function for populating the hyperparameters from the yaml file
        """
        self.batch_size = self.config["batch_size"]
        self.weight_decay = self.config["weight_decay"]
        self.momentum = self.config["momentum"]
        self.learning_rate = self.config["learning_rate"]
        self.height = self.config["image_height"]
        self.width = self.config["image_width"]
        self.num_epochs = self.config["num_epochs"]

    def build_base_graph(self):
        #TODO: Initialize your dataloader here using tf.data by calling "get_rot_data_iterator"
        self.placeholder_X = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
        self.placeholder_y = tf.compat.v1.placeholder(tf.int64, [None, 4])
        dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        dataset =  dataset.shuffle(200000).batch(self.batch_size)
        self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        data_X, data_y = self.iterator.get_next()
        logits = self.model.forward(data_X)
        #TODO: Calculate the loss and accuracy from your output logits.
        # Add your accuracy metrics and loss to the tensorboard summary using tf.summary
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=data_y, logits=logits)
        self.loss = tf.reduce_mean(entropy)

        Y_pred = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(Y_pred, axis=1)
        y_cls = tf.argmax(data_y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_cls), tf.float32))

        tf.summary.scalar('loss', data=self.loss)
        tf.summary.scalar('accuracy', data=self.accuracy)
        self.summary_op = tf.compat.v1.summary.merge_all()
        #END OF FUNCTION

    def build_train_graph(self):
        #TODO: Create an optimizer that minimizes the loss function that you defined in the above function
        optimizer = tf.compat.v1.train.AdamOptimizer()
        self.optimizer = optimizer.minimize(self.loss)

        #This will restore a model @ the latest epoch if you have already started training
        #If it cannot find a checkpoint, it will set the starting epoch to zero
        if os.path.exists("./checkpoints/model{0}".format(self.model_number)):
            #TODO: Complete the restore from checkpoint function
            self.start_epoch = self.restore_from_checkpoint()
        else:
            self.start_epoch = 0

        #Creates a writer for Tensorboard
        self.train_writer = tf.compat.v1.summary.FileWriter("./logs/train/" + str(self.model_number), self.sess.graph)

    def train(self):
        #TODO: Initialize your graph variables
        self.sess.run([tf.compat.v1.global_variables_initializer()])
        #TODO: Implement and call the get_training_data function to get the data from disk
        #NOTE: Depending on how you implement your iterator, you may not need to load the data here.
        images, labels = self.data_obj.get_training_data()
        images, labels = self.data_obj.preprocess(images)
        # #TODO: Split the data into a training and validation set: see sklearn train_test_split
        print(images.shape)
        X_train, X_val, y_train, y_val = train_test_split(images, labels)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)

        self.saver = tf.compat.v1.train.Saver()

        #TODO: Implement the training and validation loop and checkpoint your file at each epoch
        print("[INFO] Starting Training...")
        for epoch in range(self.start_epoch, self.num_epochs):
            num_batches = int(len(X_train)/self.batch_size)
            self.sess.run([self.iterator.initializer], feed_dict={self.placeholder_X: X_train, self.placeholder_y: y_train})
            for batch in range(num_batches):
                self._update_learning_rate(epoch)
                _, loss, accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy])
                #_, loss, accuracy, summary = self.sess.run([self.optimizer, self.loss, self.accuracy, self.summary_op])

                #TODO: Make sure you are using the tensorflow add_summary method to add the data for each batch to Tensorboard
                #self.train_writer.add_summary(summary, step=epoch*self.batch_size+batch)
                print("TRAINING Epoch: {0}, Batch: {1} ==> Accuracy: {2}, Loss: {3}".format(epoch, batch, accuracy, loss))

            #TODO: Calculate validation accuracy and loss
            self.sess.run(self.iterator.initializer, feed_dict={placeholder_X: X_val, placeholder_y: y_val})
            num_batches = int(len(X_val)/self.batch_size)
            for batch in range(num_batches):
                loss, accuracy = self.sess.run([self.loss, self.accuracy])
                print("VALIDATION Epoch: {0}, Batch: {1} ==> Accuracy: {2}, Loss: {3}".format(epoch, batch, accuracy, loss))
            #TODO: Use the save_checkpoint method below to save your model weights to disk.
            self.save_checkpoint(epoch*num_batches+batch, epoch)
            

        #TODO: Evaluate your data on the test set after training
        # feed_dict = {
        #     self.test_inputs : self.X_test,
        #     self.test_labels : self.y_test
        # }
        images, labels = self.data_obj.get_test_data()
        images, labels = self.data_obj.preprocess(images)
        X_test = images
        y_test = labels
        y_test = tf.keras.utils.to_categorical(y_test)
        _, accuracy = self.sess.run([self.iterator.initializer, self.accuracy], feed_dict={self.placeholder_X: X_test, self.placeholder_y: y_test})
        print("TEST Accuracy: {0}".format(accuracy))


    def predict(self, image_path):
        #TODO: Once you have trained your model, you should be able to run inference on a single image by reloading the weights
        self.saver = tf.compat.v1.train.import_meta_graph('model.meta')
        self.restore_from_checkpoint()
        img = cv2.imread(image_path)
        img = np.expand_dims(img, axis=0)
        logits = self.model.forward(img)
        pred_class = tf.argmax(tf.nn.softmax(logits), axis=1)
        return self.classes[pred_class]



    def restore_from_checkpoint(self):
        #TODO: restore the weights of the model from a given checkpoint
        #this function should return the latest epoch from training (you can get this from the name of the checkpoint file)
        checkpoint = tf.compat.v1.train.latest_checkpoint('./')
        self.saver.restore(self.sess, checkpoint)

    def save_checkpoint(self, global_step, epoch):
        #TODO: This function should save the model weights. If we are on the first epoch it should also save the graph.
        if epoch == 0:
            self.saver.save(self.sess, 'model', global_step=global_step)
        else:
            self.saver.save(self.sess, 'model', global_step=global_step, write_meta_graph=False)

    def _update_learning_rate(self, epoch):
        #In the paper the learning rate is updated after certain epochs to slow down learning.
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2