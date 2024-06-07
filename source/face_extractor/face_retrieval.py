import cv2
import numpy as np
import torch
import os 
from controlnet_aux import LineartAnimeDetector
from PIL import Image
from deepface import DeepFace


def cosine(a, b):
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean(a, b):
    return np.linalg.norm(a - b)

def preprocess_image(image):
	# Equalization Histogram
	ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

	# split color channel
	y, cr, cb = cv2.split(ycrcb_image)

	# euqalized Y for brightness
	y_equalized = cv2.equalizeHist(y)

	# merge color channel
	ycrcb_equalized = cv2.merge((y_equalized, cr, cb))

	# Transfer to RGB
	equalized_image = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

	# Denoising using  Gaussian
	denoised_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

	return denoised_image


def extract_face_with_deep_face(img,detector_backend = 'opencv'):
	face_locations = []
	try:
		face_objs = DeepFace.extract_faces(img_path = img,detector_backend = detector_backend)
	except Exception as e:
		return face_locations
	else:
		for face_obj in face_objs:
			facial_area = face_obj['facial_area']
			x = facial_area['x']
			y = facial_area['y']
			w = facial_area['w']
			h = facial_area['h']
			face_location = [y,x+w,y+h,x]
			face_locations.append(face_location)
	return face_locations



def load_lineart_map_model():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators").to(device)
	return processor

class faceRetrieval:
	def __init__(self,image_path,selected_frame,feature_extractor = 'VGG-Face',detector_backend = 'opencv',distance_metric = 'cosine',similarity_threshold_lineart = 0.3, similarity_threshold = 0.725,is_image_retrieval_face = False):
		self.image_path = image_path
		self.selected_frame = selected_frame
		self.face_retrieval = []


		self.frame_retrieval = []
		self.feature_extractor = feature_extractor
		self.similarity_threshold_lineart = similarity_threshold_lineart
		self.similarity_threshold = similarity_threshold
		self.is_image_retrieval_face = is_image_retrieval_face
		self.lineart_model = None
		self.detector_backend = detector_backend
		self.distance_metric =  distance_metric

	def feature_extractor_function(self,img1,img2):
		img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
		img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

		if self.lineart_model is None:
		 	self.lineart_model = load_lineart_map_model()
		lineart_1 = self.lineart_model(img1)
		lineart_2 = self.lineart_model(img2)

		lineart_1 = np.array(lineart_1)
		lineart_2 = np.array(lineart_2)

		lineart_1 = cv2.cvtColor(lineart_1, cv2.COLOR_RGB2BGR)
		lineart_2 = cv2.cvtColor(lineart_2, cv2.COLOR_RGB2BGR)

		result = DeepFace.verify(img1_path = lineart_1, img2_path = lineart_2, model_name = self.feature_extractor, detector_backend = 'skip', distance_metric = self.distance_metric, threshold = self.similarity_threshold_lineart)

		return result['verified']


	def find(self):
		#Read image
		img = cv2.imread(self.image_path)
		target = preprocess_image(img)

		if not self.is_image_retrieval_face:
			face_locations = extract_face_with_deep_face(target,self.detector_backend)
			for face_location in face_locations:
				top, right, bottom, left = face_location   
				face_img = img[top:bottom, left:right]
				self.face_retrieval.append(face_img)
		else:
			self.face_retrieval.append(img)
		if len(self.face_retrieval) > 0:
			self.handle_frame()
		return self.frame_retrieval

	def push_frame(self, frame, location):
		top, right, bottom, left = location
		cv2.rectangle(frame, (left, top), (right, bottom), ( 0, 255, 0), 2)
		self.frame_retrieval.append(frame)

	def handle_frame(self):
		for frame in self.selected_frame:
			frame_process = preprocess_image(frame)

			face_data_screen = extract_face_with_deep_face(frame_process,self.detector_backend)
			for i,face_location in enumerate(face_data_screen):
				top, right, bottom, left = face_location
				face_img = frame[top:bottom, left:right]

				for j,face_target in enumerate(self.face_retrieval):
					flag_lineart = self.feature_extractor_function(face_img,face_target)
					result = DeepFace.verify(img1_path = face_img, img2_path = face_target, model_name = self.feature_extractor, detector_backend = 'skip', distance_metric = self.distance_metric, threshold = self.similarity_threshold)
					if result['verified'] and flag_lineart:
						self.push_frame(frame, face_location)