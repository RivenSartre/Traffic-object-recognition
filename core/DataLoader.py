import yaml

from pathlib import Path
import glob
import cv2
import numpy as np



class DataLoder:
	def __init__(self,source_Path,stride):
		IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
		VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
		self.source_Path = source_Path
		self.videoFlag = False
		self.ImgFlag = False
		self.stride = stride
		suffix_temp = Path(source_Path).suffix.strip().lstrip('.').lower()
		self.Video = None
		self.VideoFps = None
		self.Videow = None
		self.Videoh = None
		self.frame = 0
		self.frames = 0
		if suffix_temp in IMG_FORMATS:
			self.ImgFlag = True
			self.frames = 1
		elif suffix_temp in VID_FORMATS:
			self.videoFlag = True

	def __iter__(self):

		return self

	def __next__(self):
		if self.videoFlag:
			if self.Video == None:
				self.ReadVideo(self.source_Path)
				self.VideoFps = self.Video.get(cv2.CAP_PROP_FPS)
				self.Videow = int(self.Video.get(cv2.CAP_PROP_FRAME_WIDTH))
				self.Videoh = int(self.Video.get(cv2.CAP_PROP_FRAME_HEIGHT))

			for _ in range(self.stride):
				self.Video.grab()
			rel,img0 = self.Video.retrieve()
			if rel:
				self.frame+=1
			else:
				self.Video.release()
				self.Video = None
				raise StopIteration
		else:
			if self.frame == self.frames:
				raise StopIteration
			self.VideoFps = None
			self.Videow = None
			self.Videoh = None
			img0 = cv2.imread(self.source_Path)
			self.frame+=1
		img = ImgReshape(img0)[0]  # padded resize
		img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
		img = np.ascontiguousarray(img)

		return img,img0



	def ReadVideo(self,path):
		self.frame = 0
		self.Video = cv2.VideoCapture(path)
		self.frames = int(self.Video.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)

def ImgReshape(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
				  stride=32):

	# Resize and pad image while meeting stride-multiple constraints
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better val mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	# 做边界扩充，这里的训练量输入图片好像只要比640*640小就行，后面自己写到这里看看具体效果
	# dw和dh是扩充量
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)


