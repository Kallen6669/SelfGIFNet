import os
from os import listdir
from os.path import join
import jittor as jt
import jittor.nn as nn
from args import Args as args
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

EPSILON = 1e-5

def normalize_Var(Var):
	(b, ch, h, w) = Var.shape

	Var_v = Var.view(b, -1)
	t_min = jt.min(Var_v, 1)[0]
	t_max = jt.max(Var_v, 1)[0]

	t_min = t_min.view(b, 1, 1, 1)
	t_min = t_min.repeat(1, ch, h, w)
	t_max = t_max.view(b, 1, 1, 1)
	t_max = t_max.repeat(1, ch, h, w)
	Var = (Var - t_min) / (t_max - t_min + EPSILON)
	return Var

def list_images(directory):
	images = []
	names = []
	dir = listdir(directory)
	dir.sort()
	for file in dir:
		# name = file.lower()
		name = file
		if name.endswith('.png'):
			images.append(join(directory, file))
		elif name.endswith('.jpg'):
			images.append(join(directory, file))
		elif name.endswith('.jpeg'):
			images.append(join(directory, file))
		elif name.endswith('.bmp'):
			images.append(join(directory, file))
		elif name.endswith('.tif'):
			images.append(join(directory, file))
		name1 = name.split('.')
		names.append(name1[0])
	return images, names

def gradient(x):
	dim = x.shape;
	# Jittor会自动管理设备，不需要手动指定CUDA
	# kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
	kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
	kernel = jt.array(kernel, dtype=jt.float32).unsqueeze(0).unsqueeze(0)
	kernel = kernel.repeat(dim[1], dim[1], 1, 1);
	# Jittor不需要Parameter，直接使用Var即可
	weight = kernel;
	# Jittor的ReflectionPad2d用法相同
	pad = nn.ReflectionPad2d(1);		
	# Jittor的conv2d用法与PyTorch相同
	gradMap = nn.conv2d(pad(x), weight=weight, stride=1, padding=0);
	# showTensor(gradMap);
	return gradMap;

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
	if num_imgs is None:
		num_imgs = len(image_path)
	original_imgs_path = image_path[:num_imgs]
	# random
	random.shuffle(original_imgs_path)
	mod = num_imgs % BATCH_SIZE
	print('BATCH SIZE %d.' % BATCH_SIZE)
	print('Train images number %d.' % num_imgs)
	print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		original_imgs_path = original_imgs_path[:-mod]
	batches = int(len(original_imgs_path) // BATCH_SIZE)
	return original_imgs_path, batches


def save_mat(out, path):
	# Jittor会自动管理设备，直接获取numpy数据
	out = out.data[0].numpy()
	out = np.squeeze(out)
	out = out.transpose((2, 1, 0))
	sio.savemat(path, {'img': out})


def get_image(path, height=256, width=256, flag=False):
	if flag is True:
		mode = cv2.IMREAD_COLOR
	else:
		mode = cv2.IMREAD_GRAYSCALE
	# image = Image.open(path).convert(mode)
	image = cv2.imread(path, mode)
	if height is not None and width is not None:
		# image = image.resize((height, width), Image.ANTIALIAS)
		# image = image.resize((height, width))
		image = cv2.resize(image,(height, width))
	return image


def get_train_images(paths, height=256, width=256, flag=False):
	if isinstance(paths, str):
		paths = [paths]
	images = []
	for path in paths:
		image = get_image(path, height, width, flag)
		if flag is True:
			image = np.transpose(image, (2, 0, 1))
		else:
			image = np.reshape(image, [1, image.shape[0], image.shape[1]])
		images.append(image)

	images = np.stack(images, axis=0)
	images = jt.array(images, dtype=jt.float32)
	return images

def calculate_entropy(image):
	flat_image = image.flatten()
	probabilities = np.histogram(flat_image, bins=256, range=[0, 256], density=True)[0]
	entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
	return entropy

def calculate_sd(image):
	variance = np.std(image)
	return variance

def save_metrics_to_txt(output_path, entropy, variance):
	metrics = f"{entropy}\n{variance}"

	with open(output_path[:-4]+".txt", "w") as file:
		file.write(metrics)	

def save_image(img_fusion, output_path):
	img_fusion = img_fusion.float32()
	# Jittor会自动管理设备，直接获取numpy数据
	img_fusion = img_fusion.data[0].numpy()

	img_fusion = img_fusion * 255
	img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
	if img_fusion.shape[2] == 1:
		img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])

	cv2.imwrite(output_path, img_fusion)


def show_heatmap(feature, output_path):
	sns.set()
	feature = feature.float32()
	# Jittor会自动管理设备，直接获取numpy数据
	feature = feature.data[0].numpy()

	feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + EPSILON)
	feature = feature * 255
	feature = feature.transpose(1, 2, 0).astype('uint8')
	if feature.shape[2] == 1:
		feature = feature.reshape([feature.shape[0], feature.shape[1]])

	fig = plt.figure()
	# sns.heatmap(feature, cmap='YlGnBu', xticklabels=50, yticklabels=50)
	sns.heatmap(feature, xticklabels=50, yticklabels=50)
	fig.savefig(output_path, bbox_inches='tight')
	# plt.show()


def gram_matrix(y):
	(b, ch, h, w) = y.shape
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


