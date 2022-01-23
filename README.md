# DS2L-SOM

DS2L-SOM is clustering algorithm based on Self Organizing Maps (SOM)[^1][^2].

# How to use
DSL2-SOM tries to follow the Scikit learn API. 

```python
from ds2lsom import DS2LSOM
clusterer = DS2LSOM()
labels = clusterer.fit(data).transform(data)
```

# Dependencies
- Pandas
- Numpy
- Networkx
- MiniSom
- SciKit Learn

# Notes
Currently work in process
ToDo:
- Other methods of creating prototypes (kmeans, neural gas?)
- Properly implement SOM arguments

# References
[^1]: _A Local Density-based Simultaneous Two-level Algorithm for
Topographic Clustering_, Guénaël Cabanes and Younès Bennani,
2008

[^2]: _Enriched topological learning for cluster detection and visualization_,
Guénaël Cabanes, Younès Bennani and Dominique Fresneau, 2012
