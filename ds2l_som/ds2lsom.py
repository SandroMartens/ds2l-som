from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class DS2LSOM:
    """Clustering learned from vector prototypes.

    Implementation of the paper
    "Enriched topological learning for cluster detection and visualization"
    by Guénaël Cabanes, Younès Bennani, Dominique Fresneau
    10.1016/j.neunet.2012.02.019

    Parameters
    ----------
    n_prototypes: int (optional) default = inferred from data
        Number of prototypes.

    model_args : dict of dicts (optional)
        Args passed to the vector quantization algorithm.

        "init" goes to initialization.

        "train" goes to fitting/training.

    method : {"som", "kmeans"}
        Method to compute prototypes.

    threshold : int (optional), default = 1
        Number of common samples for two prototypes to be considered connected.

        Higher: More clusters.

    sigma : num (optional), default = inferred from training data
        Bandwidth parameter for local density estimation.

        Too high: All samples influence all prototypes.

        Too low: Distant samples will not influence prototypes.

    verbose : bool, default = False
        Print information about each step.
    """

    def __init__(
        self,
        n_prototypes: Union[int, None] = None,
        threshold: int = 1,
        sigma: Union[float, None] = None,
        method: str = "som",
        verbose: bool = False,
        model_args: dict = None,
    ) -> None:

        methods = ("som", "kmeans")
        self.method = method
        if self.method not in methods:
            raise ValueError(f"{method} is not an method for prototype computation.")

        #  Update Minisom args at train time
        self.n_prototypes = n_prototypes
        self.model_args = model_args
        self.threshold = threshold
        self.sigma = sigma
        self.verbose = verbose

    def fit(self, data) -> None:
        """Fit and train SOM, enrich prototypes and return graph of prototypes.

        Parameters
        ----------
        data : array or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        self : Fitted estimator
        """
        if self.verbose:
            print("Started training.")

        sample_size = len(data)
        if self.n_prototypes is None:
            self.n_prototypes = int(10 * (sample_size ** (1 / 2)))

        self.som_dim = int((self.n_prototypes) ** (1 / 2))
        self.n_prototypes = self.som_dim**2

        self.quantizer = self._train_quantizer(data)
        # self.win_map = self.som.win_map(data, return_indices=True)
        self._get_dist_matrix(data)
        self.nbr_values, self.prototypes = self._enrich_prototypes()
        self.edge_list = self._get_edges()
        self.graph = self._create_graph()
        self._initial_clustering()
        self._final_clustering()

        if self.verbose:
            print("Training finished.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return the cluster id for each sample.

        Input
        -----
        data : array of shape (n_samples, n_features)
            Data to cluster.

        Returns
        -------
        labels : array
            Labels of clusters.
        """
        #  Get Best Matching Prototype
        if self.method == "som":
            pred = self.quantizer._distance_from_weights(data).argsort(axis=-1)[:, 0]
        elif self.method == "kmeans":
            pred = self.quantizer.predict(data)

        for sample, prototype in enumerate(pred):
            if prototype in self.graph:
                pred[sample] = self.graph.nodes[prototype]["label"]
            else:
                pred[sample] = -1
        return np.array(pred)

    def _get_dist_matrix(self, data) -> None:
        """Calculate distance matrix (i, j) for prototype i and sample j."""
        if self.method == "som":
            self.dist_matrix = self.quantizer._distance_from_weights(data).T
        elif self.method == "kmeans":
            self.dist_matrix = self.quantizer.transform(data).T

    def _train_quantizer(self, data) -> Union[MiniSom, KMeans]:
        """Define model and train on data.

        Input:
        ------
        Data, SOM args

        Returns:
        -------
        Trained SOM Object
        """
        if self.method == "som":
            minisom_args_default = {
                "init": {
                    "x": self.som_dim,
                    "y": self.som_dim,
                    "input_len": data.shape[1],
                },
                "train": {"num_iteration": 100 * len(data)},
            }

            if self.model_args is not None:
                minisom_args_default["init"].update(self.model_args["init"])
                minisom_args_default["train"].update(self.model_args["train"])

            som = MiniSom(**minisom_args_default["init"])
            som.pca_weights_init(data)
            som.train(data=data, **minisom_args_default["train"])
            self.weights = som.get_weights().reshape(-1, 2)
            return som

        if self.method == "kmeans":
            kmeans_args_default = {
                "init": {"n_clusters": self.n_prototypes},
                "train": {"sample_weight": None},
            }

            if self.model_args is not None:
                kmeans_args_default["init"].update(self.model_args["init"])
                kmeans_args_default["train"].update(self.model_args["train"])
            kmeans = KMeans(**kmeans_args_default["init"], verbose=self.verbose)
            kmeans.fit(X=data, **kmeans_args_default["train"])
            self.weights = kmeans.cluster_centers_
            return kmeans

        return None

    def _enrich_prototypes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Enrich each prototype with a local density estimate,
        a local variability estimate and connected neighbors.

        Input:
        ------
        Distance matrix Dist(w, x) between M prototypes w und the N data x

        Output:
        ------
        Dataframe with the local density D_i and local variability s_i
        associated to each prototype w_i,

        The neighborhood values v_{i,j} associated with each pair of prototypes
         w_i and w_j
        """
        self.densities = self._estimate_density()
        self.variabilities = self._estimate_local_variability()
        nbr_values = self._estimate_neighborhood_values()

        prototypes = pd.DataFrame(
            [self.densities, self.variabilities], index=["d", "s"]
        ).T
        v = pd.DataFrame(nbr_values)

        return v, prototypes

    def _estimate_density(self) -> np.ndarray:
        """Estimate local density for each prototype
        from its assigned samples.
        """
        if self.sigma is None:
            sigma = self._calculate_sigma()
        else:
            sigma = self.sigma

        #  Distances of samples where clostest prototype is prototype
        dist_matrix_sorted = self.dist_matrix.argsort(axis=0)[0]
        densities = np.zeros(shape=(self.som_dim * self.som_dim))
        for prototype in range(len(self.dist_matrix)):
            neighbors = self.dist_matrix[prototype, dist_matrix_sorted == prototype]
            neighbors = neighbors**2
            neighbors = np.e ** -(neighbors / (2 * sigma**2))
            neighbors = neighbors / sigma * np.sqrt(2 * np.pi)
            #  Surpress warning about empty slices
            if len(neighbors) > 0:
                densities[prototype] = np.mean(neighbors)
            else:
                densities[prototype] = 0

        return densities

    def _calculate_sigma(self):
        """Heuristic for sigma: Mean distance between
        prototype and nearest neighbor."""
        pairwise_prototype_distances = pairwise_distances(self.weights, self.weights)
        pairwise_prototype_distances.sort(axis=1)
        clostes_neighbor_distances = pairwise_prototype_distances[:, 1]
        sigma = np.mean(clostes_neighbor_distances)
        return sigma

    def _estimate_local_variability(self) -> np.ndarray:
        """For each prototype w, variability s is the mean distance
        between w and the L data x_w represented by w.
        """
        #  Distances of samples where clostest prototype is prototype
        dist_matrix_sorted = self.dist_matrix.argsort(axis=0)[0]
        variabilities = np.zeros(shape=(self.som_dim * self.som_dim))
        for prototype in range(len(self.dist_matrix)):
            neighbors = self.dist_matrix[prototype, dist_matrix_sorted == prototype]
            #  Surpress warning about empty slices
            if len(neighbors) > 0:
                variabilities[prototype] = np.mean(neighbors)
            else:
                variabilities[prototype] = 0

        return variabilities

    def _estimate_neighborhood_values(self) -> np.ndarray:
        """For each data x, find the two clostest prototypes
         u*(x) and u**(x) (Best Matching Units, BMUs).

        Compute the number v_{i,j} of data having i and j as first two BMUs
        """
        BMUs = np.argsort(self.dist_matrix, axis=0)[:2, :]
        v = np.zeros(shape=(self.som_dim**2, self.som_dim**2))
        u, counts = np.unique(BMUs, axis=1, return_counts=True)
        u = u.T
        for index, combination in enumerate(u):
            #  Define edge existence in both directions
            v[combination[0], combination[1]] = counts[index]
            v[combination[1], combination[0]] = counts[index]

        return v

    def _get_edges(self) -> pd.DataFrame:
        """Find list of node pairs (i, j) s.t. v_{i, j} >= threshold

        Input
        -----
            v: Matrix of connected nodes (v_{i,j} have common samples).
        Returns
        -------
            groups: Indices (source, target) of all edges
        """
        indices = np.asarray(self.nbr_values >= self.threshold).nonzero()
        edges = set(zip(indices[0], indices[1]))
        edges = pd.DataFrame(edges, columns=["source", "target"])

        return edges

    def _create_graph(self) -> nx.Graph:
        """Create Graph with edges between prototypes.

        Input:
        ------
        The list of connected prototypes,

        Data for each prototype

        Output:
        -------
        Graph object with enriched protoypes.
        """
        edges = self.edge_list
        prototypes = self.prototypes
        g = nx.from_pandas_edgelist(
            edges,
            source="source",
            target="target",
            create_using=nx.Graph,
        )

        nx.set_node_attributes(g, prototypes.d, "density")
        nx.set_node_attributes(g, prototypes.s, "variability")

        return g

    def _initial_clustering(self) -> None:
        """Label each prototype by the maximum with the highest gradient."""
        #  Create initial labels
        for node in self.graph:
            self.graph.nodes[node]["label"] = node

        #  if there are non connected components, run over each separately
        components = nx.connected_components(self.graph)
        for component in components:
            subgraph = self.graph.subgraph(nodes=component)
            longest_path = nx.diameter(subgraph)

            for i in range(longest_path):
                #  Iterate over (node, neighbors) pairs
                for node, nbr_data in subgraph.adj.items():
                    node_density = subgraph.nodes[node]["density"]
                    largest_gradient = 0
                    largest_gradient_neighbor = node
                    # iterate over neighbors of each node
                    for nbr in nbr_data:
                        nbr_density = subgraph.nodes[nbr]["density"]
                        gradient = nbr_density - node_density
                        if gradient > largest_gradient:
                            largest_gradient = gradient
                            largest_gradient_neighbor = nbr

                    self.graph.nodes[node]["label"] = self.graph.nodes[
                        largest_gradient_neighbor
                    ]["label"]

                    subgraph.nodes[node]["label"] = self.graph.nodes[
                        largest_gradient_neighbor
                    ]["label"]

    def _final_clustering(self) -> None:
        """Merge clusters according to pairwise density threshold of clusters.

        Input : Graph
        """
        converged = False
        # cont = True
        while not converged:
            # cont = False
            G = self.graph
            for e in G.edges:
                node_i = G.nodes[e[0]]
                node_j = G.nodes[e[1]]

                label_i = node_i["label"]
                label_j = node_j["label"]

                density_i = node_i["density"]
                density_j = node_j["density"]

                density_max_i = G.nodes[label_i]["density"]
                density_max_j = G.nodes[label_j]["density"]

                if density_max_i > 0 and density_max_j > 0:
                    threshold = (density_max_i**-1 + density_max_j**-1) ** -1
                else:
                    threshold = 0

                if (
                    density_i > threshold
                    and density_j > threshold
                    and label_i != label_j
                ):
                    converged = False
                    self._merge_micro_clusters(
                        G, label_i, label_j, density_max_i, density_max_j
                    )
                else:
                    converged = True
        self.graph = G

    def _merge_micro_clusters(
        self,
        G: nx.Graph,
        label_i: int,
        label_j: int,
        density_max_i: float,
        density_max_j: float,
    ) -> None:
        """Overwrite label of low density cluster with
        label of high density cluster.
        """
        if density_max_i > density_max_j:
            new_label = label_i
            old_label = label_j
        else:
            new_label = label_j
            old_label = label_i

        #  Overwrite lower density label
        for node, label in G.nodes.data("label"):
            if label == old_label:
                G.nodes[node]["label"] = new_label
