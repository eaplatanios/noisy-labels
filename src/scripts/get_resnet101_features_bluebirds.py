"""Precomputes ResNet101 features for the images."""

import os
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from noisy_ml.data.crowdsourced import BlueBirdsLoader
from noisy_ml.networks import ResNet101


def main():
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, os.pardir, "data")
    dataset = BlueBirdsLoader.load(data_dir, load_features=False)

    # Load and preprocess images.
    images = []
    for i in dataset.instances:
        i_path = os.path.join(
            data_dir,
            "crowdsourced",
            "bluebirds",
            "original_images",
            str(i) + ".jpg",
        )
        images.append(img_to_array(load_img(i_path, target_size=[224, 224])))
    images = np.stack(images)
    images = preprocess_input(images)

    # Compute ResNet101 representation.
    X = layers.Input(shape=[224, 224, 3])
    H = ResNet101(freeze=False, pooling="max", weights="imagenet")(X)
    model = models.Model(inputs=X, outputs=H)
    hiddens = model.predict(images, batch_size=64)

    # Save data.
    resnet101_features_path = os.path.join(
        data_dir, "crowdsourced", "bluebirds", "resnet101_features"
    )
    np.savez(resnet101_features_path, np.asarray(dataset.instances), hiddens)


if __name__ == "__main__":
    main()
