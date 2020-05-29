from data_loader.my_data_loader import MyDataLoader
from models.my_model import MyModel
from trainers.my_trainer import MyModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = MyDataLoader(config)

    print('Create the model.')
    model = MyModel(config)

    print('Create the trainer')
    trainer = MyModelTrainer(model.model,
                             (data_loader.get_train_data(),
                              data_loader.get_test_data()),
                             config)

    print('Start training the model.')
    trainer.train()

    
    print('Plotting random samples.')
    
    # training samples
    fig1 = plt.figure(1)
    fig1.clf()

    rand_idx = np.random.choice(data_loader.x_train.shape[0], size=4)

    y_pred = model.predict(data_loader.x_train[rand_idx,:,:,:])

    for i, idx in enumerate(rand_idx):

        img = data_loader.x_train[idx,:,:,:]

        ax = fig1.add_subplot(2,2,i+1)

        lm_x_true = data_loader.y_train[idx, 0::2]
        lm_y_true = data_loader.y_train[idx, 1::2]
        ax.imshow(np.transpose(img, axes=[1,0,2]))
        ax.plot(lm_x_true*config.data.IMAGE_SIZE,
                lm_y_true*config.data.IMAGE_SIZE, 'gx')

        lm_x_pred = y_pred[i, 0::2]
        lm_y_pred = y_pred[i, 1::2]
        ax.plot(lm_x_pred*config.data.IMAGE_SIZE,
                lm_y_pred*config.data.IMAGE_SIZE, 'rx')

        # print( np.mean((y_pred - y_train[rand_idx, :])**2) )
    fig1.savefig('training_samples.png')
        
    # test samples
    fig2 = plt.figure(2)
    fig2.clf()

    rand_idx = np.random.choice(data_loader.x_test.shape[0], size=4)

    y_pred = model.predict(data_loader.x_test[rand_idx,:,:,:])

    for i, idx in enumerate(rand_idx):

        img = data_loader.x_test[idx,:,:,:]

        ax = fig2.add_subplot(2,2,i+1)

        lm_x_true = y_test[idx, 0::2]
        lm_y_true = y_test[idx, 1::2]
        ax.imshow(np.transpose(img, axes=[1,0,2]))
        ax.plot(lm_x_true*config.data.IMAGE_SIZE,
                lm_y_true*config.data.IMAGE_SIZE, 'gx')

        lm_x_pred = y_pred[i, 0::2]
        lm_y_pred = y_pred[i, 1::2]
        ax.plot(lm_x_pred*config.data.IMAGE_SIZE,
                lm_y_pred*config.data.IMAGE_SIZE, 'rx')
        
        # print( np.mean((y_pred - y_train[rand_idx, :])**2) )
    fig2.savefig('test_samples.png')    

if __name__ == '__main__':
    main()
    
