# Cluster Based SISSO

This repository combines cluster extraction methods (Kmeans, DeepAA) with SISSO. With the former, representatives and corresponding
clusters are extracted. Then, either multitask-SISSO is trained on all materials which are not in the test set, with tasks being defined 
by the extracted clusters, or singletask-SISSO is trained on the representatives only. As both Kmeans and DeepAA are stochastic 
algorithms, they should be applied multiple times to the same dataset; median test-RMSE of subsequent sisso application then indicates the 
usefulness of the clustering algorithm with given parameters.

Installation prerequisites are:
- python>3.6
- sissopp (https://sissopp_developers.gitlab.io/sissopp/)
- tensorflow
- numpy 
- pandas 
- scipy 
- scikit-learn
- seaborn
- toml


