# 3D pose estimation for human figures from pictorial maps

This is code for the article [Inferring Implicit 3D Representations from Human Figures on Pictorial Maps](https://doi.org/10.1080/15230406.2023.2224063). Visit the [main repository](https://github.com/narrat3d/pictorial-maps-3d-humans) and the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/) for more information.  
  
It is a fork from https://github.com/una-dinosauria/3d-pose-baseline (MIT License, Copyright by Julieta Martinez, Rayat Hossain, Javier Romero).

## Usage

### Training

* Set the TRAINVAL_DATA_FOLDER in config.py to our training data.
* Scroll to the bottom of predict_3dpose.py, uncomment train() and comment eval(), select a configuration, and pass the required arguments.

### Inference

* Scroll to the bottom of predict_3dpose.py, comment train() and uncomment eval(), select a configuration, and pass the required arguments.

## Configurations

h36m = original H3.6M training and validation data
narrat3d = our own training and validation data

16j = 16 joints
21j = 21 joints

debug = uses only action "Walking" instead of "All"

cam = original four cameras with perspective projection

inf_narrat3d_val = inference with our own validation data
inf_narrat3d_test = inference with our own test data

## Notes 

* You cannot eval() directly after train() in predict_3dpose.py. You have to restart the module and comment either train() or eval().
* It is normal that a folder "-p" will be created besides the "experiments" and "normalizations" folder.
* For the test data, nan values are displayed for the metrics since there is no reference data.

© 2022-2023 ETH Zurich, Raimund Schnürer
