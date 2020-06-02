from base.base_data_loader import BaseDataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import cm


class MyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MyDataLoader, self).__init__(config)
        
        landmarks = pd.read_csv( "dataset/facial_keypoints.csv")  # load csv
        # not all images contain all landmarks
        # -> should not be a problem, but for now keep only images with all landmarks
        num_missing_landmarks = landmarks.isnull().sum( axis=1 )
        all_landmarks_present_ids = np.array(num_missing_landmarks == 0)
        
        print("num (images, landmarks*2)", landmarks.shape)
        print(landmarks.columns)
        print("images where all landmarks has been marked:",
              sum(all_landmarks_present_ids))

        d = np.load( "dataset/face_images.npz")
        dataset = d[ 'face_images' ].T
        dataset = np.reshape(dataset, (-1,
                                       self.config.data.IMAGE_SIZE,
                                       self.config.data.IMAGE_SIZE)
        )  # grayscale -> 1 component
        
        images_bw = dataset[all_landmarks_present_ids, :, :] / 255  # normalized
        images = np.reshape(images_bw,
                            (images_bw.shape[0],
                             images_bw.shape[1],
                             images_bw.shape[2],1))
        landmarks = landmarks.iloc[all_landmarks_present_ids, :]\
                             .reset_index(drop=True).values \
                             / self.config.data.IMAGE_SIZE  # normalized

        # split into train, test sets
        self.X_train, self.X_test, \
            self.y_train, self.y_test = train_test_split(
                images, landmarks,
                test_size=self.config.data.test_size
            )
        

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
