# Example Usage:
```Python
from src.no_label_pfa.execute_PFA import pfa
from src.no_label_pfa.tsne import tsne
from src.no_label_pfa.umap import umap
from src.no_label_pfa.dbscan import dbscan
from src.no_label_pfa.hdbscan import hdbscan
from src.no_label_pfa.find_cluster_differences import find_cluster_differences
from src.no_label_pfa.compare_dbscan_labels import compare_dbscan_labels
from src.no_label_pfa.split_data import split_data
from src.no_label_pfa.get_mutual_information import get_mutual_information
from src.no_label_pfa.validate_feature_selection import validate_feature_selection
from src.no_label_pfa.shaply_explanation import shaply_explanation
from src.no_label_pfa.tree_explanation import tree_explanation

if __name__ == "__main__":
  path_original_data="path/to/my/no_label_file.csv"
  
  pfa(path=path_original_data)
  
  umap(path_original_data=path_original_data, n_neighbors=15)
  
  dbscan("umap_output.csv", eps=1, min_samples=15)
  # or optionally:
  hdbscan("umap_output.csv", min_cluster_size=15, plot=False)
  
  compare_dbscan_labels("comparison_labels.csv")

  # optional:
  split_data(path_original_data, n_splits=5)
  
  find_cluster_differences(path_original_data=path_original_data,clusters=[0,1])
  
  get_mutual_information(path_original_data,clusters=[0,1])

  validate_feature_selection(path_original_data,clusters=[0,1])

  shaply_explanation(path_original_data, n_highest_mutual_information=10, clusters=[0,1])
  # or
  tree_explanation(path_original_data, n_highest_mutual_information=10, min_samples_leaf=50, clusters=[0,1])

```
