import os
import torch
import cv2
from ultralytics import YOLO
from typing import Optional, Union, List
from PIL import Image
import numpy as np

class Pipeline:
	def __init__(
			self,
			yolo_path: str,
			device: Optional[str]=None
	):
		"""
		:param yolo_path: path to the finetunned YOLO model
		"""
		self.yolo = None
		self.load_yolo(yolo_path)

		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			print(f"Using device {self.device}")
		else:
			self.device = device

	def load_yolo(
			self,
			yolo_path: str
	):
		"""
		Load a YOLO model from a specified path.

		:param yolo_path: Path to the YOLO model
		"""
		if os.path.exists(yolo_path):
			try:
				self.yolo = YOLO(yolo_path)
			except Exception as e:
				raise Exception(f"Error loading YOLO model: {e}")
		else:
			self.yolo = None
			print(f"YOLO model not found at {yolo_path}. Train a YOLO model before running the pipeline.")

	def train_yolo(
			self,
			data_yaml: str,
			weights: str="yolo11n.pt",
			save_path: str="yolo11n_finetuned.pt",
			img_size: int=640,
			epochs: int=50
	):
		"""
		Train YOLO with a specified dataset.

		:param data_yaml: Path to the data.yaml file (specifying paths to train/val and class info)
		:param weights: Path to the pretrained weights (default: yolov11n.pt)
		:param img_size: Image size
		:param batch_size: Batch size
		:param epochs: Number of epochs for training
		"""
		print(f"Starting YOLOv11 training with {epochs} epochs on device {self.device}")
		model = YOLO(weights).to(self.device)
		model.train(
			data=data_yaml,
			imgsz=img_size,
			epochs=epochs,
			device=self.device
		)
		model.save(save_path)
		print("Training Complete.")
	
	def detect(
			self,
			source: Union[str, Image.Image, List[Image.Image]],
			img_size: int=640,
			conf_thresh: int=0.25,
			max_det: int=-1
	) -> List[List[dict]]:
		"""
		Function to run inference using the trained YOLO model.

		:param source: Path, image or list of images to run inference on
		:param img_size: Image size for inference
		:param conf_thresh: Confidence threshold
		:param max_det: Maximum detections per image
		"""
		# Run inference
		results = self.yolo(source, imgsz=img_size)

		# Extract image, bounding boxes and confidence
		boxes = [
			[
				{
					"image": Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
					"box": box.cpu().numpy().astype(int).tolist(), # [x1, y1, x2, y2]
					"conf": conf.item(),
				}
				for img, box, conf in zip(result.orig_img, result.boxes.xyxy, result.boxes.conf)
				if conf.item() >= conf_thresh				
			]
			for result in results
		]

		# Keep only the top max_det detections
		if max_det > 0:
			boxes = [b[:max_det] for b in boxes]

		return boxes

	def extract_boxes(
			self,
			source: List[List[dict]]
	) -> List[List[Image.Image]]:
		"""
		Extract the crops from the images based on the bounding boxes.

		:param source: Output of the detect function
		"""
		crops = [
			[
				box["image"].crop(box["box"])
				for box in boxes
			]
			for boxes in source
		]
		return crops

	def segment(
			self,
			source: List[List[Image.Image]]
	) -> List[List[List[Image.Image]]]:
		"""
		Segment the license plates into individual characters.

		:param source: Output of the extract_boxes function
		"""
		segments = []
		for img in source:
			img_segments = []
			for plate in img:
				plate_segments = []
				# Convert to grayscale if not already
				if plate.mode != "L":
					gray = plate.convert("L")
				else:
					gray = plate.copy()
				gray = np.array(gray)
				# Preprocess
				blurred = cv2.GaussianBlur(gray, (5, 5), 0)
				_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
				# Find contours
				contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # also cv2.RETR_EXTERNAL
				# Get bounding boxes of contours
				bounding_boxes = []
				for contour in contours:
					x, y, w, h = cv2.boundingRect(contour)
					aspect_ratio = w / float(h)
					if 0.2 < aspect_ratio < 1.0: # Filter
						bounding_boxes.append((x, y, w, h))
				# Sort from left to right
				bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
				# Extract segments
				for x, y, w, h in bounding_boxes:
					segment = plate.crop((x, y, x+w, y+h))
					plate_segments.append(segment)
				img_segments.append(plate_segments)
			segments.append(img_segments)
		return segments
	
	def __call__(
			self,
			source: Union[str, Image.Image, List[Image.Image]],
			img_size: int=640,
			conf_thresh: int=0.25,
			max_det: int=-1
	) -> List[List[List[Image.Image]]]:
		"""
		Run the entire pipeline.

		:param source: Path, image or list of images to run inference on
		:param img_size: Image size for inference
		:param conf_thresh: Confidence threshold
		:param max_det: Maximum detections per image
		"""
		detections = self.detect(source, img_size, conf_thresh, max_det)
		crops = self.extract_boxes(detections)
		segments = self.segment(crops)
		return segments
