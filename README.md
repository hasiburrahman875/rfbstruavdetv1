
# This is the repo for the implementation of "Airborne Object Detection from Moving Drones with Multiple Receptive Field Blocks and Swin Transformer" _submitted in IROS-25_

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

## 4. Configure Dataset Paths
Modify the corresponding YAML file to specify the training and testing folder paths. Update the file located at:
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

**Note:**
- For the FL-Drone dataset, the image size (`--img`) should be at most **640**.
- For other datasets, the image size can be **1280, 800, or 640**.

