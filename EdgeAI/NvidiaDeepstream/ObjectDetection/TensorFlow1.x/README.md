## How to install TensorFlow1.15.4

1. Install system packages required by TensorFlow

   ```
   sudo apt-get update
   sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
   ```

2. Install and Upgrade Pip3

   ```
   sudo apt-get install python3-pip
   sudo pip3 install -U pip testresources setuptools==49.6.0
   ```

3. Install the Python package dependencies.

   ```
   sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
   sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
   ```

4. Assign Jetpack Version and Install TensorFlow
   <br>
   I'm using Deepstream5.0, so I'm assigning Jetpack 4.4

   ```
   sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'
   ```

   **NOTE**<br>
   Referred Documentation [HERE](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)。
