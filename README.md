# Demo

## Dependencies
Part of the needed dependencies are listed as below for the experiment.
```
python 3.5.4
h5py 2.7.0
keras 2.3.1
numpy 1.18.2
pandas 0.20.3
pillow 4.2.1
scipy 1.2.1
tensorflow-cpu 2.2.0
```
To install on Linux or MacOS：
```
Anaconda is used to manage different development environments for deep learning. 

Based on Anaconda, users can create the above environment through the following instructions:

source activate
conda create -n deversitySQ python=3.5.4
conda activate deversitySQ
pip install tensorflow==2.2.0
pip install keras==2.3.1
pip install numpy==1.18.2
pip install pandas==0.20.3
pip install pillow==4.2.1
pip install scipy==1.2.1
pip install h5py==2.7.0
```

## Usage
```
Main function for creating a diversity seed queue

positional arguments:
  seed                the random seed
  {True,False}        whether the test cases are stored in classification
  categories_number   the numbers of selected test cases from each category
  test_suite_dir      the dir that saves the test cases
  candidate_set_size  the size of candidate set
  {True,False}        whether some selected seeds are forgotten
  forgetting_number   the number of forgetting selected seeds
  target_dir          the dir that saves the selected seeds
  {cosin,L1,L2,Lmax}  the method for calculating the distance between two test
                      cases

optional arguments:
  -h, --help          show this help message and exit
```

## File structure
- createSQ.py: the main interface
- randomMethod.py: randomly selects seeds from test suite
- informationBasedMethod.py: selects seeds from test suite according to IBM-SQ strategy
- DNNBasedMethod.py: selects seeds from test suite according to FB-SQ strategy with ART*
- DNNBasedMethodForgetting.py selects seeds from test suite according to FB-SQ strategy with ART*_{forgetting}