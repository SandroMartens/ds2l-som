import numpy as np
import networkx as nx
import pandas as pd
from minisom import MiniSom
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Union

"""Implementation of the paper
"Enriched topological learning for cluster detection and visualization" by
Guénaël Cabanes, Younès Bennani, Dominique Fresneau
10.1016/j.neunet.2012.02.019
"""


class DS2LSOM:
    """Clustering learned from SOM prototypes.

    Parameters
    ----------
    minisom_args : dict of dicts (optional)
        Args passed to MiniSom.

        "init"  : Initialize SOM.

        "train" : Training args.

    method : string {"som", "kmeans"}, default: "som"
        Method to compute prototypes.

    threshold : int (optional, default = 1)
        Number of common samples for two prototypes to be considered connected.

        Higher: More clusters.

    sigma : num (optional, default = inferred from training data)
        Bandwidth parameter for local density estimation.

        Too high: All samples influence all prototypes.

        Too low: Distant samples will not influence prototypes.
    """
    def __init__(self,
                 minisom_args: dict = None,
                 threshold: int = 1,
                 sigma: float = None,
                 method: str = "som"
        ) -> None:

        methods = ("som", "kmeans")
        self.method = method
        if self.method not in methods:
            raise ValueError(f"{method} is not an method for prototype computation.")

        #  Update Minisom args at train time
        self.minisom_args = minisom_args
        self.threshold = threshold
        self.sigma = sigma

    def fit(self, data):
        """Fit and train SOM, enrich prototypes and return graph of prototypes.

        Parameters
        ----------
        data : array or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        self : Fitted estimator
        """
        sample_size = len(data)
        self.n_prototypes = int(10 * (sample_size ** (1 / 2)))
        self.som_dim = int((self.n_prototypes) ** (1 / 2))
        # self.som_sigma = 0.1 * self.som_dim
        minisom_args = {
            "x": self.som_dim,
            "y": self.som_dim,
            "sigma": 1,
            "input_len": data.shape[1],
        }

        if self.minisom_args is not None:
            minisom_args.update(self.minisom_args)

        self.som = self._get_prototypes(data, minisom_args)
        self.win_map = self.som.win_map(data, return_indices=True)
        self._get_dist_matrix(data)
        self.nbr_values, self.prototypes = self._enrich_prototypes()
        self.edge_list = self._get_edges()
        self.graph = self._create_graph()
        self._initial_clustering()
        self._final_clustering()

        return self

    def _get_dist_matrix(self, data):
        if self.method == "som":
            self.dist_matrix = self.som._distance_from_weights(data).T
        elif self.method == "kmeans":
            self.dist_matrix = self.som.transform(data).T

    def predict(self, data) -> np.ndarray:
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
        if self.method == "som":
            y = self._predict_som(data)
        elif self.method == "kmeans":
            y = self.som.predict(data)
        return y

    def _predict_som(self, data) -> np.ndarray:
        win_map = self.som.win_map(data, return_indices=True)
        graph = self.graph
        pred = dict()
        for prototype_index, samples_indices in win_map.items():
            index_flat = np.ravel_multi_index(
                prototype_index, (self.som_dim, self.som_dim)
            )
            if index_flat in self.graph:
                for sample in samples_indices:
                    pred[sample] = graph.nodes[index_flat]["label"]
            else:
                for sample in samples_indices:
                    pred[sample] = -1

            y = pd.DataFrame(pred.items(), columns=["sample", "label"])
            y = y.sort_values("sample").label
        return np.array(y)

    def _get_prototypes(self, data, minisom_args:dict) -> Union[MiniSom,KMeans]:
        """Define model and train on data.

        Input:
        ------
        Data, SOM args

        Returns:
        -------
        Trained SOM Object
        """
        if self.method == "som":
            som = MiniSom(**minisom_args)
            som.pca_weights_init(data)
            som.train(data=data, num_iteration=20_000)
            return som

        elif self.method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_prototypes)
            kmeans.fit(data)
            return kmeans

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
        #  Heuristic for sigma: Mean distance between
        #  prototype and nearest sample.
        if self.sigma is None:
            self.sigma = np.nanmean(self.dist_matrix.min(axis=1))

        indices = self.win_map
        densities = np.zeros(shape=(self.som_dim, self.som_dim))

        for prototype_index, samples_indices in indices.items():
            index_flat = np.ravel_multi_index(
                prototype_index, (self.som_dim, self.som_dim)
            )
            neighbors = self.dist_matrix[index_flat, samples_indices]
            neighbors = neighbors ** 2
            neighbors = np.e ** -(neighbors / (2 * self.sigma ** 2))
            neighbors = neighbors / self.sigma * np.sqrt(2 * np.pi)
            neighbors = np.mean(neighbors)

            densities[prototype_index] = np.mean(neighbors)

        return densities.flatten()

    def _estimate_local_variability(self) -> np.ndarray:
        """For each prototype w, variability s is the mean distance
        between w and the L data x_w represented by w.
        """
        indices = self.win_map
        variabilities = np.zeros(shape=(self.som_dim, self.som_dim))

        for prototype_index, samples_indices in indices.items():
            index_flat = np.ravel_multi_index(
                prototype_index, (self.som_dim, self.som_dim)
            )
            variabilities[prototype_index] = self.dist_matrix[
                index_flat, samples_indices
            ].mean()

        return variabilities.flatten()

    def _estimate_neighborhood_values(self) -> np.ndarray:
        """For each data x, find the two clostest prototypes
         u*(x) and u**(x) (Best Matching Units, BMUs).

        Compute the number v_{i,j} of data having i and j as first two BMUs
        """
        BMUs = np.argsort(self.dist_matrix, axis=0)[:2, :]
        v = np.zeros(shape=(self.som_dim ** 2, self.som_dim ** 2))
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
        groups = {index for index in zip(indices[0], indices[1])}
        groups = pd.DataFrame(groups, columns=["source", "target"])

        return groups

    def _create_graph(self) -> nx.DiGraph:
        """Create Graph with edges between prototypes. Edges are directed from
        high density nodes to low density nodes with a positive gradient.

        Input:
        ------
        The list of connected prototypes,

        Data for each prototype

        Output:
        -------
        Graph object with protoypes and gradients between prototypes
        """
        edges = self.edge_list
        #  Filter out prototypes without samples
        prototypes = self.prototypes[self.prototypes["d"] > 0]
        prototypes = self.prototypes
        for i in range(len(edges)):
            edges.loc[i, "gradient"] = (prototypes.d[edges.loc[i, "target"]] -
                                        prototypes.d[edges.loc[i, "source"]])

        positive_edges = edges[edges.gradient > 0]
        g = nx.from_pandas_edgelist(
            positive_edges,
            source="source",
            target="target",
            edge_attr="gradient",
            create_using=nx.DiGraph,
        )

        nx.set_node_attributes(g, prototypes.d, "density")
        nx.set_node_attributes(g, prototypes.s, "variability")

        return g

    def _initial_clustering(self) -> None:
        """Label each prototype by the maximum with the highest gradient."""
        #  Create initial labels
        for node in self.graph:
            self.graph.nodes[node]["label"] = node

        #  Number of needed iterations
        longest_path = nx.algorithms.dag.dag_longest_path_length(self.graph)

        for i in range(longest_path):
            #  Iterate over (node, neighbor) pairs
            for node, edges in self.graph.pred.items():

                #  get largest gradient neighbor
                largest_gradient = 0
                largest_gradient_neighbor = node
                for edge in edges.items():
                    current_neighbor = edge[0]
                    current_gradient = edge[1]["gradient"]

                    if current_gradient > largest_gradient:
                        largest_gradient = current_gradient
                        largest_gradient_neighbor = current_neighbor

                self.graph.nodes[node]["label"] = self.graph.nodes[
                    largest_gradient_neighbor
                ]["label"]

    def _final_clustering(self) -> None:
        """Merge clusters according to pairwise density threshold of clusters.

        Input : Graph
        """
        cont = True
        while cont:
            cont = False
            G = self.graph
            for e in G.edges:
                node_i = G.nodes[e[0]]
                node_j = G.nodes[e[1]]

                label_i = node_i["label"]
                label_j = node_j["label"]

                density_i = node_i["density"]
                density_j = node_j["density"]

                density_max_i = node_i["density"]
                density_max_j = node_j["density"]


                threshold = (1/density_max_i + 1/density_max_j) ** -1
                # if (density_max_i == 0 or density_j == 0):
                #     logging.warning(f"Density error: {density_max_i, density_max_j} Threshold: {threshold}")

                if (
                    density_i > threshold
                    and density_j > threshold
                    and label_i != label_j
                ):
                    cont = True
                    self._merge_micro_clusters(
                        G, label_i, label_j, density_i, density_j
                    )
        self.graph = G

    def _merge_micro_clusters(self, 
            G: nx.DiGraph, 
            label_i: int, 
            label_j: int, 
            density_i: float, 
            density_j: float
        ) -> None:
        """Overwrite label of low density cluster with
        label of high density cluster.
        """
        if density_i > density_j:
            new_label = label_i
            old_label = label_j
        else:
            new_label = label_j
            old_label = label_i

        #  Overwrite lower density label
        for node, label in G.nodes.data("label"):
            if label == old_label:
                G.nodes[node]["label"] = new_label
