from ds2lsom import DS2LSOM
import numpy as np
from sklearn.datasets import load_digits

mnist_X = load_digits().images.reshape(1797, -1)
test = DS2LSOM(n_prototypes=200)
test.fit(mnist_X)
test.predict(mnist_X)
