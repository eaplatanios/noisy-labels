"""Precomputes VGG16 features for the images."""

import os
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from noisy_ml.data.crowdsourced import BlueBirdsLoader
from noisy_ml.networks import VGG16

def main():
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  dataset = BlueBirdsLoader.load(data_dir, load_features=False)

  images = []
  for i in sorted(dataset.instances):
    i_path = os.path.join(
      data_dir, "crowdsourced", "bluebirds", "resized_images", str(i) + ".jpg")
    images.append(
      img_to_array(load_img(i_path, target_size=[224, 224])))
  images = np.stack(images)
  images = preprocess_input(images)

  X = layers.Input(shape=[224, 244, 3])
  H = VGG16(freeze=False, pooling='max', weights='imagenet')(X)
  model = models.Model(inputs=X, outputs=H)
  model.compile('sgd')

  hiddens = model.predict(images, batch_size=64)
  print(hiddens.shape)
  np.save(os.path.join(data_dir, "crowdsourced", "bluebirds", "vgg16_features.npy"), hiddens)

if __name__ == '__main__':
  main()
