## Feature matching with RANSAC

Note: work-in-progress

Todo:
- given features and matches, estimate {translation, affine, homography, ...} model while using RANSAC to help mitigate incorrect matches
- add visualization(s) for inliers and outliers

This repo is a framework for modeling the transform between two images by detecting and matching the features in those images. The pipeline 
consists of 3 steps:
- detect features in each image
- match features between a pair of images
- model the transform of the matched features

The scripts folder contains numerous visualization utilities. Run in the following order, progress from writing and testing the building 
blocks to actual results can be seen:

Visualizations of the models
- [images and features](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_image_and_features.py)
![image](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/images/image_and_features.jpg)
- [feature matches](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_feature_matches.py)
- [apply translation](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_apply_translation.py)
- [apply affine](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_apply_affine.py)
- [apply homography](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_apply_homography.py)

Visualization of the algorithms
- [detect features](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_detect_features.py)
- [match features](https://github.com/merrillmckee/feature_matching_with_ransac/blob/main/scripts/visualize/run_visualize_match_features.py)
- todo: model transform
