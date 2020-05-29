from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense

class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()

    def build_model(self):

        # we use a pre-trained model as basis for the landmark detector
        # we do not load fully connected output layers at the end
        # we specify the shape of our input data (VGG16 is made for 224x224x3)
        base_input = Input(shape=(self.config.data.IMAGE_SIZE,
                                  self.config.data.IMAGE_SIZE, 3))  # must be 3 channel
        base_model = VGG16(input_tensor=base_input, include_top=False)
        # mark loaded layers as not trainable, note how summary changes
        for layer in base_model.layers:
            layer.trainable = False
            
        # additional layers, flatten->dense->output
        flat1 = Flatten()(base_model.output)
        dense1 = Dense(1024, activation='relu')(flat1)
        output = Dense(self.config.data.NUM_LANDMARKS, activation=None)(dense1)
        # define the new model
        self.model = Model(inputs=base_input, outputs=output)

        # define loss metric as mean squared error because this is a regression problem
        self.model.compile( loss=tf.keras.losses.mean_squared_error,
                            optimizer=self.config.model.optimizer,
                            metrics=['mse'] )
