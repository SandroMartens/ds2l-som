# DS2L-SOM

DS2L-SOM is clustering algorithm based on Self Organizing Maps (SOM)[^1][^2].

# How to use
DSL2-SOM follows the scikit-learn API. 

```python
from ds2lsom import DS2LSOM
clusterer = DS2LSOM()
labels = clusterer.fit(data).predict(data)
```

# Dependencies
- Pandas
- Numpy
- NetworkX
- MiniSom
- scikit-learn

# Notes
ToDo:
- Other methods of creating prototypes (vector quantization, neural gas)
- Examples

[^1]: _A Local Density-based Simultaneous Two-level Algorithm for
Topographic Clustering_, Guénaël Cabanes and Younès Bennani,
2008

[^2]: _Enriched topological learning for cluster detection and visualization_,
Guénaël Cabanes, Younès Bennani and Dominique Fresneau, 2012
