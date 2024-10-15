# ANPR-GIA

## Set up
1. Create a virtual environment  
```bash
conda env create -f environment.yml
conda activate yolo
```
2. Install pytorch  
	For macOS:  
	`conda install pytorch::pytorch torchvision -c pytorch`  
	For Windows or Linux, if you have Nvidia GPU:  
	`conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia`  
	For Linux, if you have AMD GPU:  
	`pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0`  
	If not, install for CPU:  
	`conda install pytorch torchvision cpuonly -c pytorch`  

## Repo structure
- [`pipeline.py`](pipeline.py): **contains the full and clean implementation of our pipeline to detect and recognize license plates.**
- [`notebooks/`](notebooks/): folder with notebooks of the process that lead to the final code.
	- [`comparisonOCR.ipynb`](notebooks/comparisonOCR.ipynb): script with metrics comparing Tesseract, EasyOCR and PaddleOCR for recognition.
	- [`comparisonYoloMathMorph.ipynb`](notebooks/comparisonYoloMathMorph.ipynb): script with examples and metrics of the two detection models we used.
	- [`customOCR.ipynb`](notebooks/customOCR.ipynb): an attempt to train a Convolutional Neural Net on EMNIST
	- [`evaluation.ipynb`](notebooks/evaluation.ipynb): **an example of use of our pipeline, along with its evaluation on the test set.**
	- [`generate_dataset_recognition.ipynb`](notebooks/generate_dataset_recognition.ipynb): script to generate a syntetic dataset of license plates to recognize.
	- [`mathMorph.ipynb`](notebooks/mathMorph.ipynb): the process followed to implement a mathematical morphology method to detect license plates.
	- [`segmentation.ipynb`](notebooks/segmentation.ipynb): evaluation of only the segmentation step of our pipeline.
	- [`yolo11.ipynb`](notebooks/yolo11.ipynb): training of the YOLOv11 detection model and example inference.
- [`Models/`](Models/): folder containing the trained `yolo11n_licenseplates.pt`, which can also be found [on HuggingFace](https://huggingface.co/Pikurrot/yolo11n-licenseplates).
