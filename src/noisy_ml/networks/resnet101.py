"""VGG networks."""

import tensorflow as tf

from tensorflow.keras import applications, models


def ResNet101(freeze=True, pooling="max", weights="imagenet"):
    """ResNet101 base network."""

    def network(inputs):
        # Build the base
        net = applications.ResNet101(
            include_top=False,
            input_tensor=inputs,
            pooling=pooling,
            weights=weights,
        )
        features = models.Model(
            inputs=net.input, outputs=net.get_layer(f"{pooling}_pool").output
        )

        if freeze:
            return tf.stop_gradient(features.output)
        else:
            return features.output

    return network
