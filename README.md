# DS2L-SOM

DS2L-SOM is clustering algorithm based on Self Organizing Maps (SOM)[1][2].

# How to use
DSL2-SOM tries to follow the Scikit learn API. 

```python
from dsl2som import DSL2SOM
clusterer = DSL2SOM()
labels = clusterer.fit(data).transform(data)
```

# Dependencies
- Pandas
- Numpy
- Networkx
- MiniSom
- SciKit Learn

# Notes
Currently work in process.

# References
[1] A Local Density-based Simultaneous Two-level Algorithm for
Topographic Clustering, Guénaël Cabanes and Younès Bennani,
2008

[2] Enriched topological learning for cluster detection and visualization
Guénaël Cabanes, Younès Bennani and Dominique Fresneau, 2012
