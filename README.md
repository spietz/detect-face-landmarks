# Use

Ensure that consistent versions of the various packages required for
the code are used by generating a virtual environment and installing
pip packages (python3) listed in
[requirements.txt](https://github.com/spietz/Tensorflow-Project-Signs/blob/master/requirements.txt):
```
pip3 install -r requirements.txt
```

Download face images with marked landmarks dataset at
https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points/
and store them in "datasets" in the root directory.

# Acknowledgements

* Blog post on face landmark detection:
https://towardsdatascience.com/face-landmark-detection-with-cnns-tensorflow-cf4d191d2f0

* Blog post on transfer learning:
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

# TODO
* Make it work... accuracy < 1e-4.
* Create loss, accuracy summaries.
* Retrain entire model, the VGG16 weights may not be good for grayscale input.
* Try other base models.

