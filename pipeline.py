import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import Levenshtein
from ultralytics import YOLO
from typing import Optional, Union, List
from PIL import Image
from paddleocr import PaddleOCR
from collections import defaultdict

def avg(lst):
	return sum(lst) / len(lst)

def accuracy_score(gt: str, pred: str) -> int:
	return int(gt == pred)

def ANLS(predictions: str, gt: str) -> float:
	d = Levenshtein.distance(predictions, gt)
	max_len = max(len(predictions), len(gt))
	if max_len > 0:
		return 1 - d / max_len
	else:
		return 1.0
	
def plot_confusion_matrix(matrix: np.ndarray, characters: str):
	matrix = matrix.astype(int)
	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=characters, yticklabels=characters)
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Ground Truth')
	ax.set_title('Confusion Matrix')
	plt.show()
	
characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
char_to_index = {char: idx for idx, char in enumerate(characters)}
matrix_size = len(characters)

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
		self.ocr_reader = PaddleOCR(lang="en")

		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			print(f"Using device {self.device}")
		else:
			self.device = device

		self.cache = {}

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
					"path": result.path,
					"image": Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)),
					"box": box.cpu().numpy().astype(int).tolist(), # [x1, y1, x2, y2]
					"conf": conf.item(),
				}
				for box, conf in zip(result.boxes.xyxy, result.boxes.conf)
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
	) -> List[List[dict]]:
		"""
		Extract the crops from the images based on the bounding boxes.

		:param source: Output of the detect function
		"""
		crops = [
			[
				{
					"path": box["path"],
					"box": box["image"].crop(box["box"])
				}
				for box in boxes
			]
			for boxes in source
		]
		return crops

	def segment(
			self,
			source: List[List[dict]]
	) -> List[List[dict]]:
		"""
		Segment the license plates into individual characters.

		:param source: Output of the extract_boxes function
		"""
		segments = []
		for img in source:
			img_segments = []
			for plate in img:
				path, plate = plate.values()
				plate_segments = []

				# Resize the plate to a constant size
				plate = plate.resize((200, 50))
				
				# Convert to grayscale if not already
				if plate.mode != "L":
					gray = plate.convert("L")
				else:
					gray = plate.copy()
				gray = np.array(gray)
				
				# Binarize (black on white blackground)
				blurred = cv2.GaussianBlur(gray, (5, 5), 0)
				thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
				
				# Invert (white on black background)
				thresh = cv2.bitwise_not(thresh)
				
				# Find contours
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
				hierarchy = hierarchy[0]

				bounding_boxes = []
				for i, contour in enumerate(contours):
					# Obtain bounding boxes
					x, y, w, h = cv2.boundingRect(contour)

					# Filter boxes that can't be characters
					aspect_ratio = w / float(h)
					area = w * h
					parent_idx = hierarchy[i][3]
					child_idx = hierarchy[i][2]
					if 0.1 < aspect_ratio < 2.0 and 70 < area < 800: # Has the right shape and size
						is_valid_character = True
						# Check if is not a hole (for 6, 8, 9, 0)
						# Holes' boxes are children of other bigger boxes
						if parent_idx != -1:
							continue
						if child_idx != -1:
							while child_idx != -1:
								child_area = cv2.contourArea(contours[child_idx])
								if child_area > 0.5 * area:
									is_valid_character = False
									break
								child_idx = hierarchy[child_idx][0]
						
						if is_valid_character:
							bounding_boxes.append((x, y, w, h))
				
				# Sort boxes from left to right
				bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

				# Extract segments, preprocess and filter again
				for i, (x, y, w, h) in enumerate(bounding_boxes):
					# Crop the segment
					segment = plate.crop((x, y, x+w, y+h))

					# Resize to a constant size
					new_w, new_h = 60, 80
					segment = segment.resize((new_w-20, new_h-20))
					segment_rgb = np.array(segment)

					# Binarize
					segment = segment.convert("L")
					_, segment = cv2.threshold(np.array(segment), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
					segment = Image.fromarray(segment)

					# Place segment on a larger white background
					background = Image.new('L', (new_w, new_h), 255)
					offset = (10, 10)
					background.paste(segment, offset)
					segment = background

					# Filter blue segments (left side of the plate)
					segment_hsv = cv2.cvtColor(segment_rgb, cv2.COLOR_RGB2HSV)
					lower_blue = np.array([100, 50, 50])
					upper_blue = np.array([140, 255, 255])
					blue_mask = cv2.inRange(segment_hsv, lower_blue, upper_blue)
					blue_pixel_count = cv2.countNonZero(blue_mask)
					total_pixel_count = segment_hsv.shape[0] * segment_hsv.shape[1]
					blue_ratio = blue_pixel_count / total_pixel_count
					is_blue = blue_ratio > 0.5

					# Filter blobs
					eroded = 255 - np.array(segment)
					eroded = cv2.erode(eroded, np.ones((3, 3)), iterations=8)
					is_blob = np.count_nonzero(eroded) / (new_w*new_h) > 0.05
					
					if not is_blue and not is_blob:
						plate_segments.append(segment)

				img_segments.append({"path": path, "segments": plate_segments})
			segments.append(img_segments)
		return segments
	
	def recognize(
			self,
			source: List[List[dict]],
			thresh: float=0.5
	):
		"""
		Recognize the characters in the segments.

		:param source: Output of the segment function
		"""
		recognized = []
		for img in source:
			img_recognized = []
			for plate in img:
				path, segments = plate.values()
				plate_recognized = []
				
				# Run OCR on each segment
				results = [
					self.ocr_reader.ocr(
						np.array(segment),
						det=False,
						rec=True,
						cls=False
					)[0]
					for segment in segments
				]

				# Extract the recognized text and confidence
				for result in results:
					texts = [text for text, conf in result]
					confs = [conf for text, conf in result]
					argmax = np.argmax(confs)
					text = texts[argmax]
					conf = confs[argmax]
					if conf > thresh: # Filter by confidence
						plate_recognized.append({"text": text, "conf": conf})
						
				img_recognized.append({"path": path, "plate": plate_recognized})
			recognized.append(img_recognized)
		return recognized
	
	def format_result(
			self,
			source: List[List[dict]],
			conf: bool=True,
			img: bool=True,
			gt: bool=True,
			detections: Optional[List[List[dict]]]=None,
	) -> List[List[dict]]:
		"""
		Format the result into the desired format.

		:param source: Output of the recognize function
		:param conf: Whether to include confidence in the result
		:param img: Whether to include the image in the result
		:param gt: Whether to include the ground truth in the result
		:param detections: Output of the detect function
		"""
		joined = []
		for img_idx, img in enumerate(source):
			img_joined = {"plates": []}
			for plate_idx, plate in enumerate(img):
				path, plate = plate.values()
				# Remove all non-alphanumeric and lowercase characters
				plate_text = "".join([char["text"].upper() for char in plate
							if char["text"].isalnum()])
				
				# Calculate average confidence
				avg_conf = np.mean([char["conf"] for char in plate]) if plate else 0

				# Get bounding box if detections are provided
				box = detections[img_idx][plate_idx]["box"] if detections else None

				plate_info = {"text": plate_text}
				if conf:
					plate_info["conf"] = avg_conf
				if box:
					plate_info["box"] = box
				if img and "image" not in img_joined:
					img_joined["image"] = Image.open(path)
				if gt and "gt" not in img_joined:
					img_joined["gt"] = path.split("/")[-1].split(".")[0][:7]
				img_joined["plates"].append(plate_info)
			joined.append(img_joined)
		return joined

		
	def __call__(
			self,
			source: Union[str, Image.Image, List[Image.Image]],
			img_size: int=640,
			det_thresh: float=0.25,
			max_det: int=-1,
			rec_thresh: float=0.2,
			**kwargs
	) -> List[List[dict]]:
		"""
		Run the entire pipeline.

		:param source: Path, image or list of images to run inference on
		:param img_size: Image size for inference
		:param conf_thresh: Confidence threshold for detection
		:param max_det: Maximum detections per image
		:param rec_thresh: Confidence threshold for OCR
		"""
		cache_process = kwargs.get("cache_process", False)
		return_box = kwargs.get("return_box", True)
		return_conf = kwargs.get("return_conf", True)
		return_img = kwargs.get("return_img", True)
		return_gt = kwargs.get("return_gt", True)

		# Pipeline
		detections = self.detect(source, img_size, det_thresh, max_det)
		crops = self.extract_boxes(detections)
		segments = self.segment(crops)
		recognized = self.recognize(segments, rec_thresh)

		# Format result
		result = self.format_result(
			recognized,
			conf=return_conf,
			img=return_img,
			gt=return_gt,
			detections=detections if return_box else None
		)

		if cache_process:
			# Cache the results
			self.cache = {
				"detections": detections,
				"crops": crops,
				"segments": segments,
				"recognized": recognized,
				"result": result
			}
		return result
	
	def plot_result(self, result: dict):
		"""
		Plot the results for a single image.

		:param result: A single item from the output of format_result
		"""
		# Extract the information
		assert "image" in result, "Image not found in the result"
		img_rgb = result["image"].convert("RGB")
		gt = result.get("gt", "Unknown")
		plates = result["plates"]

		fig, ax = plt.subplots(1)
		ax.imshow(np.array(img_rgb))
		ax.set_title(gt, fontsize=16)

		for plate_info in plates:
			text = plate_info["text"]
			conf = f"{plate_info['conf']:.2f}"
			box = plate_info.get("box", None)

			if box:
				# Draw the bounding box
				x_min, y_min, x_max, y_max = box
				rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
										linewidth=2, edgecolor='red', facecolor='none')
				ax.add_patch(rect)

				# Calculate the position for the text
				text_x = (x_min + x_max) / 2
				text_y = y_min - 0.03 * img_rgb.height

				# Write the recognized text
				ax.text(text_x, text_y, f"{text} {conf}%", color='black', 
						fontsize=10, ha='center', backgroundcolor='white')

		plt.axis('off')
		plt.show()

	def evaluate(
			self,
			results: dict
	) -> dict:
		"""
		Evaluate the results against the ground truth.

		:param results: The output of format_result
		"""
		assert all(["gt" in result for result in results]), "Ground truth not found in the result"
		
		# Keep the most similar prediction in each image by ANLS
		anls = []
		preds = []
		confs = []
		gt = []
		for result in results:
			res_plates = result["plates"]
			res_anls = [ANLS(plate["text"], result["gt"]) for plate in res_plates]
			argmax = np.argmax(res_anls)
			anls.append(res_anls[argmax])
			preds.append(res_plates[argmax]["text"])
			confs.append(res_plates[argmax]["conf"])
			gt.append(result["gt"])

		# Accuracy
		accuracy = [accuracy_score(p, g) for p, g in zip(preds, gt)]

		# Confusion matrix
		confusion_matrix = np.zeros((matrix_size, matrix_size))
		for p, g in zip(preds, gt):
			for i, (p_char, g_char) in enumerate(zip(p, g)):
				confusion_matrix[char_to_index[g_char]][char_to_index[p_char]] += 1
		
		# Character Precision, Recall, F1 (micro-averaging)
		TP = sum(confusion_matrix[i][i] for i in range(matrix_size))
		FP = sum(confusion_matrix[:, i].sum() - confusion_matrix[i][i] for i in range(matrix_size))
		FN = sum(confusion_matrix[i, :].sum() - confusion_matrix[i][i] for i in range(matrix_size))

		precision = TP / (TP + FP) if (TP + FP) != 0 else 0
		recall = TP / (TP + FN) if (TP + FN) != 0 else 0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

		# Most confused characters
		most_confused_chars = defaultdict(int)
		for i in range(matrix_size):
			for j in range(matrix_size):
				if i != j:
					most_confused_chars[(characters[i], characters[j])] = confusion_matrix[i][j]
		most_confused_chars = sorted(most_confused_chars.items(), key=lambda x: x[1], reverse=True)

		return {
			"accuracy": accuracy,
			"anls": anls,
			"confs": confs,
			"avg_accuracy": avg(accuracy),
			"avg_anls": avg(anls),
			"avg_conf": avg(confs),
			"conf_matrix": confusion_matrix,
			"char_precision": precision,
			"char_recall": recall,
			"char_f1": f1,
			"most_confused_chars": most_confused_chars
		}
		
