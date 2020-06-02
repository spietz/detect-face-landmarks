from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense

class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()
        self.model.summary()
        
    def build_model(self):

        self.model = tf.keras.Sequential(
            layers = [
                Conv2D(256, input_shape=(self.config.data.IMAGE_SIZE,
                                         self.config.data.IMAGE_SIZE, 1),
                       kernel_size=(3, 3), strides=2, activation='relu'),
                Conv2D(256, kernel_size=(3, 3), strides=2, activation='relu'),
                BatchNormalization(),

                Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
                Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
                BatchNormalization(),

                Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
                Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
                BatchNormalization(),

                Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
                Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
                BatchNormalization(),

                Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu'),
                Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu'),
                BatchNormalization(),

                # additional layers, flatten->dense->output
                Flatten(),
                Dense(1024, activation='relu'),
                Dense(self.config.data.NUM_LANDMARKS, activation=None),
            ]
        )

        if self.config.model.checkpoint:
            self.model.load_weights(self.config.model.checkpoint)
        
        # define loss metric as mean squared error because this is a regression problem
        self.model.compile( loss=tf.keras.losses.mean_squared_error,
                            optimizer=self.config.model.optimizer,
                            metrics=['mse'] )
