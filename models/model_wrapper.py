import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from models import model

def model_wrapper(wts_path, train=False, to_save_as=False, model_path=None):
    # If a pre-trained model is specified, load it and return
    if model_path:
        return tf.keras.models.load_model(model_path)

    # Create a new model using the defined architecture
    my_model = model.get_model()

    # If pre-trained weights are specified, load them into the model
    if wts_path:
        my_model.load_weights(wts_path)

    # If training is requested
    if train:
        # Define a custom callback to stop training if both training and validation accuracy are above 95%
        class myCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
                    print('Stopping training')
                    my_model.stop_training = True

        # Instantiate the custom callback
        callbacks = myCallback()

        # Load the MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the pixel values to be between 0 and 1
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Train the model for 10 epochs with the custom callback
        my_model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

        # Evaluate the model on the test set and print the results
        print(my_model.evaluate(x_test, y_test))

        # Save the trained weights with a timestamp
        if wts_path:
            my_model.save_weights('{}-{}'.format(wts_path, round(time.time())))
        else:
            my_model.save_weights(to_save_as)

    # Return the trained or pre-trained model
    return my_model
