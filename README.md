# kaggle-fisheries-steller-sea-lion
Our solution for Kaggle competetion NOAA Fisheries Steller Sea Lion count, big thanks to the authors of all kernels & posts, which were of great inspiration

## Methodology
(Charles)

## Experiments
Various experiments were tried in order to solve this problem, a breif overview is given below with the respective notebooks.

 1. **Segmentation**: Segmentation is applied to the image to predict small squares centered on the coordinates of the dots as shown in the image below. [Tiramisu](https://arxiv.org/abs/1611.09326) network is used with 1:1 upsampling to predict 6 classes(5 types of sea lions and background), Log loss is used with Nadam optimizer for training. Results with this approach were not satisfactory, the model was biased towards predicting the background.An experiment with camvid dataset is shown in the notebook `[experiment_daft/Tiramisu.ipynb](https://github.com/syeddanish41/kaggle-fisheries-steller-sea-lion/blob/master/experiment-daft/Tiramisu.ipynb)`

 
 2. **Masking with segmentation**: A mask is created to focus the predictor on the areas where sea lions might be present using the same segmentation approach as above by increasing the size of the squares for sea lions and using the dice coefficient loss, it was able to provide satisfactory results in detecting the regions of interest but after running a regression it seems to overfit a lot, due to lack of time more experiments were not performed with this approach. An example is shown for predicting ROI

 
 3. **FCN Regression**: Thanks to [@mrgloom](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/33900
) we were able to get into top 100 using the fully connected net to perform regression on the whole image, an average of our two best-performing models give us our current position the leaderboard.
 
 ## Requirements
 - Python 3.5 or above
 - Jupyter 4.3.1
 - Keras 2 or above
 - Tensorflow 1.0 or above
 
 ## License
 This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/syeddanish41/kaggle-fisheries-steller-sea-lion/blob/master/LICENSE) file for details
