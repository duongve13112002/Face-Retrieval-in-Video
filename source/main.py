import cv2
import numpy as np
import os
import torch
from key_frame_extractor.key_frame import FrameExtractor
from key_frame_extractor.cluster_key_frame import KeyFrameSelector
from face_extractor.face_retrieval import faceRetrieval
import argparse


def save_img(images,folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
	for count, image in enumerate(images):
		cv2.imwrite(os.path.join(folder, str(count)+'.png'),image)

#args.query, args.video, args.output, args.is_face, args.len_slide_window, args.max_frame_one_time, args.type_window, args.threshold, args.threshold_lineart
def Extraxting_face(image_path, video_path, output, is_image_retrieval_face, len_slide_window,  max_frame_one_time, type_window, similarity_threshold, similarity_threshold_lineart,detector_backend,feature_extractor,metric):
	#hyperparameters
	video_path = video_path
	image_path = image_path
	is_image_retrieval_face = is_image_retrieval_face
	similarity_threshold_lineart = similarity_threshold_lineart
	similarity_threshold = similarity_threshold
	len_slide_window = len_slide_window
	max_frame_one_time = max_frame_one_time
	type_window = type_window

	current_directory = os.getcwd()

	#Select frame according to motion
	process_video = FrameExtractor(len_slide_window = len_slide_window, max_frame_one_time = max_frame_one_time, type_window = type_window)
	selected_frame = process_video.extract_key_frame(video_path)

	#Clustering selected_frame
	final_images = KeyFrameSelector()
	selected_frame = final_images.select_best_frames(selected_frame,None)


	#Finding similar faces in video

	face_retrieval = faceRetrieval(image_path, selected_frame,feature_extractor,detector_backend,metric,similarity_threshold_lineart,similarity_threshold,is_image_retrieval_face)

	#Screens have similar image's face
	screen_face = face_retrieval.find()
	save_img(screen_face,output)

def main():
	parser = argparse.ArgumentParser(description="Face Retrieval in Video.")
	parser.add_argument("--query", type=str, required=True, help="Path to the query image.")
	parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
	parser.add_argument("--output", type=str, required=True, help="Path for frame output.")
	parser.add_argument("--is_face", type=bool, default=False, help="Is a face query image?")
	parser.add_argument("--len_slide_window", type=int, default=10, help="The length of sliding window.")
	parser.add_argument("--max_frame_one_time", type=int, default=480, help="A number of frames are handled at one time.")
	parser.add_argument("--type_window", type=str, default='hanning', help="The type of sliding window (flat, hanning, hamming, bartlett, blackman).")
	parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for face recognition.")
	parser.add_argument("--threshold_lineart", type=float, default=0.3, help="Confidence threshold for lineart.")
	parser.add_argument("--detector_backend", type=str, default='retinaface', help="Model for detecting faces (opencv, ssd, dlib, mtcnn, fastmtcnn, retinaface, mediapipe, yolov8, yunet, centerface).")
	parser.add_argument("--feature_extractor", type=str, default='VGG-Face', help="Feature Extractor for face (VGG-Face, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet).")
	parser.add_argument("--metric", type=str, default='cosine', help="Metric for comparing faces (cosine, euclidean, euclidean_l2).")

	args = parser.parse_args()

	merge_models(args.query, args.video, args.output, args.is_face, args.len_slide_window, args.max_frame_one_time, args.type_window, args.threshold, args.threshold_lineart, args.detector_backend, args.feature_extractor, args.metric)

if __name__ == '__main__':
	main()