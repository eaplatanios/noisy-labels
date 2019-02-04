"""VGG networks."""

import tensorflow as tf

from tensorflow.keras import applications, models


def VGG16(
    freeze=True,
    pooling='max',
    weights='imagenet'):
  """VGG16 base network."""
  def network(inputs):
    # Build the base
    vgg16 = applications.vgg16.VGG16(
      include_top=True,
      input_tensor=inputs,
      pooling=pooling,
      weights=weights)
    features = models.Model(
      inputs=vgg16.input,
      outputs=vgg16.get_layer('fc2').output)

    if freeze:
      return tf.stop_gradient(features.output)
    else:
      return features.output

  return network
