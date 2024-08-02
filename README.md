# Project Page

A project page for our IJCAI 2023 paper "Contact2Grasp: 3D Grasp Synthesis via Hand-Object Contact Constraint".

# Copyright

## training the Param2Mesh
1. setting the args.use_model=“Param2Mesh”
2. setting the ContactPose dataset dir
```
train_pkl_name="contactpose_fixed_train_replace.pkl"
test_pkl_name="contactpose_fixed_test_replace.pkl"
``` 
#### Download MANO Model
Download the Python 3-compatible MANO code from the [manopth website](https://github.com/hassony2/manopth). Copy the `mano` folder from the manopth project to the root of the ContactOpt folder.

Due to license restrictions, the MANO data files must be downloaded from the original [project website](https://mano.is.tue.mpg.de/). Create an account and download 'Models & Code'. Extract the `models` folder to the recently created `mano` directory. The directory structure should be arranged so the following files can be found:
```
mano/webuser/lbs.py
mano/models/MANO_RIGHT.pkl
```

