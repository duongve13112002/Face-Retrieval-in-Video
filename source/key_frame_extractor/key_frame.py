import cv2
import os
import operator
import numpy as np
from scipy.signal import argrelextrema
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class FrameExtractor(object):
	def __init__(self,len_slide_window=10,max_frame_one_time = 480,type_window = 'hanning'):
		self.len_slide_window = len_slide_window
		self.max_frame_one_time = max_frame_one_time
		self.type_window = type_window

	def calculate_difference_of_frames(
		self,
		frame: Optional[np.ndarray],
		prev_frame: Optional[np.ndarray]):
		if frame is not None and prev_frame is not None:
			frame_HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
			pre_frame_HSV = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2HSV)

			diff = cv2.absdiff(frame_HSV,pre_frame_HSV)
			count = np.sum(diff)

			return count
		return None

	def extract_frames_from_video(self,videopath: str):
		cap = cv2.VideoCapture(videopath)

		ret, frame = cap.read()
		while ret:
			prev_frame = None
			frame_diffs = []
			frames = []
			for _ in range(0,self.max_frame_one_time):
				if ret:
					diff_pre_frame = self.calculate_difference_of_frames(frame,prev_frame)
					if diff_pre_frame is not None:
						frame_diffs.append(diff_pre_frame)
						frames.append(frame)
					prev_frame = frame
					ret, frame = cap.read()
				else:
					cap.release()
					break
			yield frames, frame_diffs
		cap.release()

	def __smooth__(
		self,
		x : Optional[np.ndarray]
		) -> np.ndarray:
		
		if x.ndim != 1:
			raise (ValueError, "smooth only accepts 1 dimension arrays.")

		if x.size < self.len_slide_window:
			raise (ValueError, "Input vector needs to be bigger than window size.")

		if self.len_slide_window < 3:
			return x

		if not self.type_window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
			raise (
				ValueError,
				"Smoothing Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
			)

		s = np.r_[2 * x[0] - x[self.len_slide_window:1:-1], x, 2 * x[-1] - x[-1:-self.len_slide_window:-1]]

		#moving average
		if self.type_window == "flat":
			w = np.ones(self.len_slide_window,"d")
		else:
			w = getattr(np, self.type_window)(self.len_slide_window)

		y = np.convolve(w / w.sum(), s, mode="same")
		return y[self.len_slide_window - 1 : -self.len_slide_window + 1]

	def get_frames(
		self,
		frames:Optional[List],
		frame_diffs: Optional[List]) -> Optional[List]:

		key_frames = []
		diff_arr = np.array(frame_diffs)

		sm_diff_array = self.__smooth__(diff_arr)

		frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

		for frame_index in frame_indexes:
			key_frames.append(frames[frame_index - 1])
		return key_frames


	def extract_key_frame(self,videopath) -> Optional[List] :
		key_frames = []

		frame_extractor_from_video_generator = self.extract_frames_from_video(videopath)


		for frames, frame_diffs in frame_extractor_from_video_generator:
			key_frames_chunk = []
			key_frames_chunk = self.get_frames(frames,frame_diffs)
			key_frames.extend(key_frames_chunk)
		return key_frames

	def save_frame_to_storage(
		self,
		frame : np.ndarray,
		file_path :str,
		file_name:str,
		file_ext:str
		) -> None:

		file_full_path = os.path.join(file_path, file_name + file_ext)
		cv2.imwrite(file_full_path, frame)