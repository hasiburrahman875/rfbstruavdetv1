
# This is the repo for the implementation of "Airborne Object Detection from Moving Drones with Multiple Receptive Field Blocks and Swin Transformer" _submitted in IROS-25_

# RFB-SPATIAL Implementation

# Validation Instructions

## 1. Clone the Repository
Clone the repository to your local machine using the following command:
```bash
git clone <repository_url>
cd <repository_name>
```

## 2. Set Up Python Environment
Create a Python environment and install the required dependencies from `requirements.txt` inside the `rfb-spatial` folder:
```bash
cd rfb-spatial
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Download Datasets and Weights
Follow the dataset download instructions to obtain the necessary datasets and pre-trained weights. 

**Dataset Link:** https://mailmissouri-my.sharepoint.com/:f:/r/personal/mrpk9_umsystem_edu/Documents/Reserach%20Synch/IROS/Dataset?csf=1&web=1&e=CEcHBp

**Weights Link:** https://mailmissouri-my.sharepoint.com/:f:/r/personal/mrpk9_umsystem_edu/Documents/Reserach%20Synch/IROS/Weights?csf=1&web=1&e=elEXit

## 4. Configure Dataset Paths
Modify the corresponding YAML file to specify the training and testing folder paths, number of classes, and class names. Update the file located at:
```bash
data/dataset_name.yaml
```
Set the correct paths for your dataset.

## 5. Run Validation
Activate the Python environment and execute the following command in the terminal:
```bash
python val.py --data data/dataset_name.yaml --weights path_to_the_weight --img provide_img_size --batch 1 --name experiment_name
```
Ensure that the paths and parameters are correctly provided before running the command.

# Training Instructions

## 1. Download and configure the dataset path mentioned above

## 2. Run Training
Activate the Python environment and execute the following command in the terminal:
```bash
python train.py --data data/dataset_name.yaml --hyp data/hyps/hyp.UAVDT.yaml --img 1280 --device 0,1 --batch 8 --cfg models/rf-aod.yaml --epoch 300 --adam --name experiment_name 
```
Ensure that the paths and parameters are correctly provided before running the command.

