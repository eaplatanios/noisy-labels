"""Model utility functions."""

import logging
import numpy as np
import tensorflow as tf

__all__ = ["log1mexp"]

logger = logging.getLogger(__name__)


def log1mexp(x, name=None):
    """Computes `log(1 - exp(x))` using a numerically
  stable approach.

  We follow the approach shown in Equation 7 of:
  https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
  """
    with tf.name_scope(name, "log1mexp", values=[x]):
        x = tf.convert_to_tensor(x, name="x")

        threshold = np.log(2)
        is_too_small = tf.less(-x, threshold)
        too_small_value = tf.log(-tf.expm1(x))

        # This `where` will ultimately be a no-op because we
        # will not select this code-path whenever we use the
        # surrogate `ones_like`.
        x = tf.where(is_too_small, -tf.ones_like(x), x)

        y = tf.log1p(-tf.exp(x))
        return tf.where(is_too_small, too_small_value, y)
