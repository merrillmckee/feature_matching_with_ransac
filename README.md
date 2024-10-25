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
[images and features]()
[feature matches]()
[apply translation]()
[apply affine]()
[apply homography]()

Visualization of the algorithms
[detect features]()
[match features]()
todo: [model transform]()
