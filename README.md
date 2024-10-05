# ANPR-GIA

## Set up
1. Create a virtual environment  
```bash
conda env create -f environment.yml
conda activate yolo
```
2. Install pytorch
	For maxOS:  
	`conda install pytorch::pytorch==2.0.0 torchaudio==2.0.0 -c pytorch`  
	For Windows or Linux, if you have Nvidia GPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`  
	For Linux, if you have AMD GPU:  
	`pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/rocm6.0`  
	If not, install for CPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 cpuonly -c pytorch`  

## References
- https://edatos.consorciomadrono.es/dataset.xhtml?persistentId=doi:10.21950/OS5W4Z
