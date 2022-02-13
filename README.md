# DS2L-SOM

DS2L-SOM is clustering algorithm based on Self Organizing Maps (SOM).

# How to use
## Installing
```bash 
git clone https://github.com/SandroMartens/ds2l-som.git
cd ds2l_som
pip install -e .
```

## In a project
DSL2-SOM follows the scikit-learn API. We can train on data in the form `(n_samples, n_features)`.

```python
from ds2lsom import DS2LSOM
clusterer = DS2LSOM()
clusterer.fit(data)
labels = clusterer.predict(data)
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

# References
- _A Local Density-based Simultaneous Two-level Algorithm for
Topographic Clustering_, Guénaël Cabanes and Younès Bennani,
2008
- _Enriched topological learning for cluster detection and visualization_,
Guénaël Cabanes, Younès Bennani and Dominique Fresneau, 2012
