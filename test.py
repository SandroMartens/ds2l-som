from topologicalLearningClusterer import TopologicalLearningClustering
import numpy as np
from sklearn.datasets import load_digits

mnist_X = load_digits().images.reshape(1797, -1)
test = TopologicalLearningClustering()
test.fit(mnist_X)
test.predict(mnist_X)
