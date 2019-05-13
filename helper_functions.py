import torch
import torch.optim as optim
from torchvision import transforms, models

import requests
from PIL import Image
from io import BytesIO


def vgg19(pretrained=True, requires_grad=False, only_features=True):
	'''Returns vgg19 model with some options to be customized'''

	model = models.vgg19(pretrained=pretrained)

	if not requires_grad:
		for param in model.parameters():
			param.requires_grad_(False)

	if only_features:
		return model.features
	else:
		return model


def load_image(path, max_size=256, shape=None):
	'''Load in and transform an image 
	   with max_size not larger than specified
	   or a given shape (if specified)
    '''

	if 'http' in path:
		respone = requests.get(path)
		image = Image.open(BytesIO(response.content)).convert('RGB')
	else:
		image = Image.open(path).convert('RGB')

	if shape is not None:
		size = shape
	else:
		# resize height and width to be smaller than max_size
		h, w = image.shape[-2:]
		if h >= max(w, max_size):
			h, w = (max_size, w * max_size / h)
		elif w >  max(h, max_size):
			h, w = (h * max_size / w, max_size)
		size = (h, w)

	transform = transforms.Compose([
		transforms.Resize(size),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		])

	# remove the transparent (alpha) channel and add the batch (singleton) dimension
	image = transform(image)[:3,:,:].unsqueeze(0)
	
	return image


def convert_image(tensor):
	'''Converts image from a Tensor image to a NumPy image'''

	image = tensor.to('cpu').clone().detach()
	image = image.numpy().squeeze() # remove single-dimensional entries from the shape of an array.
	image = image.transpose(1, 2, 0)
	image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
	image = image.clip(0, 1) # just in case that some values lies ooutside the [0, 1] range

	return image