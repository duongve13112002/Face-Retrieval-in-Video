import os
import cv2
import numpy as np
import hdbscan
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class KeyFrameSelector(object):

	def __getstate__(self):
		self_dict = self.__dict__.copy()
		return self_dict

	def __setstate__(self, state):
		self.__dict__.update(state)

	def variance_of_laplacian(self,image):
		return cv2.Laplacian(image, cv2.CV_64F).var()

	def get_laplacian_score(
		self,
		images: List,
		n_images:List) -> List:
		
		variance_laplacians = []
		for image_i in n_images:
			img_file = images[n_images[image_i]]
			img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

			variance_laplacian = self.variance_of_laplacian(img)
			variance_laplacians.append(variance_laplacian)

		return variance_laplacians



	def clustering_with_hdbscan(self,images):
		all_dst = []

		for image in images:
			img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (256, 256))
			imf = np.float32(img) / 255.0
			dst = cv2.dct(imf)  # the dct
			dst = dst[:16, :16]
			dst = dst.reshape((-1))
			all_dst.append(dst)

		Hdbascan = hdbscan.HDBSCAN(min_cluster_size=3,metric='euclidean').fit(all_dst)
		labels = np.add(Hdbascan.labels_,1)
		nb_clusters = len(np.unique(Hdbascan.labels_))

		clusters_index_array = []
		clusters_index_array_of_only_one_image = []

		for i in np.arange(nb_clusters):
			if i == 0:
				index_array = np.where(labels == i)
				clusters_index_array_of_only_one_image.append(index_array)
			else:
				index_array = np.where(labels == i)
				clusters_index_array.append(index_array)

		return clusters_index_array,clusters_index_array_of_only_one_image

	#returns one best image from each cluster
	def get_best_image_from_each_cluster(self,images, clusters_index_array):
		filtered_images = []

		clusters = np.arange(len(clusters_index_array))
		for i in clusters:
			curr_row = clusters_index_array[i][0]
			n_images = np.arange(len(curr_row))

			variance_laplacians = self.get_laplacian_score(images, n_images)

			try:
				selected_frame_of_current_cluster = curr_row[np.argmax(variance_laplacians)]
				filtered_images.append(selected_frame_of_current_cluster)
			except:
				break

		return filtered_images

	def select_best_frames(
		self, 
		input_key_frames : Optional[List[np.ndarray]],
		output_folder: Optional[str] = None) -> Optional[List]:
		
		filtered_images_list = []
		i = 0

		if not isinstance(input_key_frames,list):
			input_key_frames = [input_key_frames]
		if len(input_key_frames) >=1:
			clusters_index_array,clusters_index_array_of_only_one_image = self.clustering_with_hdbscan(input_key_frames)
			selected_images_index = self.get_best_image_from_each_cluster(input_key_frames,clusters_index_array)

			clusters_index_array_of_only_one_image = [item for t in clusters_index_array_of_only_one_image for item in t]
			clusters_index_array_of_only_one_image = clusters_index_array_of_only_one_image[0].tolist()
			selected_images_index.extend(clusters_index_array_of_only_one_image)
			for index in selected_images_index:
				img = input_key_frames[index]
				filtered_images_list.append(img)

			if output_folder is not None and output_folder =="":
				for images in clusters_index_array:
					path = output_folder + '/' + str(i)
					try:
						if not os.path.isdir(output_folder):
							os.mkdir(output_folder)
					except OSError:
						print ("Creation of the directory %s failed" % output_folder)

					try:
						os.makedirs(path)
					except:
						pass
					for image in images[0]:
						cv2.imwrite(os.path.join(path, str(image)+'.jpeg'),input_key_frames[image])
					i +=1

				for image in clusters_index_array_of_only_one_image:
					path = output_folder+'/'+str(i)
					try:
						if not os.path.isdir(output_folder):
							os.mkdir(output_folder)
					except OSError:
						print ("Creation of the directory %s failed" % output_folder)

					try:
						os.makedirs(path)
					except:
						pass
					cv2.imwrite(os.path.join(path, str(image)+'.jpeg'),input_key_frames[image])
					i +=1
		else:
			for img in input_key_frames:
			 	filtered_images_list.append(img)
		return filtered_images_list