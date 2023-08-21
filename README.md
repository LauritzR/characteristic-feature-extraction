If you use the presented code or the provided Python scripts inspired you for further extensions or variations of this framework, we’ll be happy if you cite our paper “” () in course of which the Python implementations of this git repository have been worked out.

# Example Usage:
```Python
from src.characteristic_feature_extraction.execute_PFA import pfa
from src.characteristic_feature_extraction.tsne import tsne
from src.characteristic_feature_extraction.umap import umap
from src.characteristic_feature_extraction.dbscan import dbscan
from src.characteristic_feature_extraction.hdbscan import hdbscan
from src.characteristic_feature_extraction.find_cluster_differences import find_cluster_differences
from src.characteristic_feature_extraction.compare_dbscan_labels import compare_dbscan_labels
from src.characteristic_feature_extraction.split_data import split_data
from src.characteristic_feature_extraction.get_mutual_information import get_mutual_information
from src.characteristic_feature_extraction.validate_feature_selection import validate_feature_selection
from src.characteristic_feature_extraction.shaply_explanation import shaply_explanation
from src.characteristic_feature_extraction.tree_explanation import tree_explanation

if __name__ == "__main__":
  
  data = pd.read_csv("path/to/my/no_label_file.csv", sep=',', header=None)

  pfa(data)
  
  umap(data, n_neighbors=15)
  
  dbscan("umap_output.csv", eps=1, min_samples=15)
  # or optionally:
  hdbscan("umap_output.csv", min_cluster_size=15, plot=False)
  
  compare_dbscan_labels("comparison_labels.csv")

  # optional:
  split_data(data, n_splits=5)
  
  find_cluster_differences(data,clusters=[0,1])
  
  get_mutual_information(data,clusters=[0,1])

  validate_feature_selection(data,clusters=[0,1])

  shaply_explanation(data, n_highest_mutual_information=10, clusters=[0,1])
  # or
  tree_explanation(data, n_highest_mutual_information=10, min_samples_leaf=50, clusters=[0,1])

```


:exclamation: As of June 2023, the UMAP method did not work with numba versions > 0.56.4. Implementation tested with numba version == 0.56.4 :exclamation:
