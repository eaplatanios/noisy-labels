"""VGG networks."""

import tensorflow as tf

from tensorflow.keras import applications


def VGG16(
    freeze=True,
    pooling='max',
    weights='imagenet'):
  """VGG16 base network."""
  def network(inputs):
    # Build the base
    network = applications.vgg16.VGG16(
      include_top=False,
      input_tensor=inputs,
      pooling=pooling,
      weights=weights)

    if freeze:
      return tf.stop_gradient(network.output)
    else:
      return network.output

  return network
