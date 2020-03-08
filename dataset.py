from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from utils import is_image_file, load_img

noise_count = 5000
noise_range = 40;
def add_noise(img):
  # print(img.size,np.asarray(img).shape,'img')
  image_size = 286
  noiseimg = np.zeros((3, image_size, image_size), dtype=np.float32)
  # prepare a noise image
  for ii in range(noise_count):
      xx = random.randrange(image_size)
      yy = random.randrange(image_size)

      noiseimg[0][yy][xx] += random.randrange(-noise_range, noise_range)
      noiseimg[1][yy][xx] += random.randrange(-noise_range, noise_range)
      noiseimg[2][yy][xx] += random.randrange(-noise_range, noise_range)

  x = noiseimg.astype(float)
  y = np.asarray(img).astype(float)
  y = np.moveaxis(y, 2, 0) 
  # print(x.shape,y.shape)
  result = np.clip(x+y,0,255).astype('uint8')
  # print(np.max((result)),np.min(x+y))
  # result = np.moveaxis(y, 0, -1) 
  # print(result.shape)
  # result = Image.fromarray(result)
  result = np.moveaxis(result, 0, -1) 
  return result

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.aug_transform = transforms.Compose([
            transforms.Lambda(lambda x: add_noise(x) ),
          ])

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        
        b_noisy = self.aug_transform(b)
        # print(type(a),type(b),type(b_noisy))

        # print(a.size,b.size,b_noisy.shape)
        
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        b_noisy = transforms.ToTensor()(b_noisy)

        # print('tensors',a.size(),b.size(),b_noisy.size())

        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b_noisy = b_noisy[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        b_noisy = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b_noisy)
        
        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)
            b_noisy = b_noisy.index_select(2, idx)

        if self.direction == "a2b":
            return a, b, b_noisy
        else:
            return b, b_noisy, a

    def __len__(self):
        return len(self.image_filenames)
