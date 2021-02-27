### To Do


# There are two places where a distance metric can be used. One for the algorithm itself,
# the other for the silhouette score calculation, where we need a distance measure again for
# the intra and inter distance calculation.  
# I believe it should be meaningful to use the same metric in both places, but it is worth checking.
# So here are the issues. Tslearn Kmeans clustering extends sklearns kmeans,a and a dtw distance metric is added, and they also provide a sihouette score calculator based on dtw as well, which is very useful.
# Sklearn does not support both not for k-means and for silhouette, so it should be logical that yellow-bricks's silhouette visuallizer, which is based on sklearns metrics, also does not support
#dtw. But the thing is the implementation of silhoutte visualizer is so bad from their doc, that they do not take not only the model's distance metric by default, but also do not allow to specify a custom distance metric. So no matter what model you pass (with any distance metric), the sihouette visualzier always returns the score with euclidean distance (btw the results are the same as l2).
https://github.com/DistrictDataLabs/yellowbrick/blob/master/yellowbrick/cluster/silhouette.py
# So here what needs to be done.
# Check if it is logical to use different metrics for clustering and silhouette score calculations. Such as dtw for clustering and euclidean for silhouette, which would allow to visualize using silhouette visualizer.
# It might be worth extending Silhouette visualizer to be able to specify at least the other metrics by sklearn, and not call with the default parameters.
# Thirdly, it might be worth creating a custom silhouette visualizer based on dynamic time-warping itself, which would be an interesting contribution and allow to visualize.



## I think a silhouette visualzier based on dtw should be created and to make the analysis 
## more complete, use a few distance metric variations. This would allow to use at least two 
## metrics, dtw and euclidean and visualize, because tslearn's kmeans only support three metrics.
## Prior to this it  should be worth checking whether it makes sense to 1 metric for clustering and another for silhoueete, but I would not mix up for now, this would introduce a lot of possibilites.

