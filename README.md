# ImageProcessingLi
An image processing project for lithium metal/ NMC particles - streamline the data analysis project for R&D labs and university labs.


## Introduction

This analysis notebook works for JEOL SEM images, gather the length scale from the image and show the interquartile distribution for particle sizes. However, the notebook still requires some human intervention such as inputting the length scale or determining the optimal number of clusters in a k-means clustering operation. 


## Order of operations
The order of operations can be seen in the main.ipynb file
1. Load images
2. Crop the white background (frequent occurence when screenshotting images)
3. Crop the Region of Interest (ROI)
4. Gather length-scale information from the original image (input needed)
5. Determine optimal number of clusters using silhouette or elbow analysis (silhouette preferred)
6. Quantize the image and calculate histogram
7. Find particle size distribution

## Explanation on some of the function choices

Quantization refers to the process of reducing the number of distinct colors (or shades) by mapping a large, continuous set to a smaller, discrete set. The quantization algorithm used in this project is kmeans clustering. Information on kmeans clustering function of cv2 can be accessed here: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

Quantization is necessary here to improve the thresholding automation accuracy. As an example, here are the two histogram of unquantized and quantized images:

Images go here

As you can see, it will be quite difficult to automate the thresholding selection process (which will ultimately become an input in the contouring operation) with an unquantized image. Here are the final results of the area that are chosen to be analyzed - for otsu thresholding of unquantized image and for the thresholding method that we developed for quantized image:

Images go here

For the quantization process. To choose the optimal amount of clusters, we used silhouette analysis. I prefer this because it's easier to quantify which are the best choices compared to the somewhat qualitative method of elbow analysis. The silhouette analysis used in this project is based off of the simplified silhouette defined in https://en.wikipedia.org/wiki/Silhouette_(clustering)


## Final results

Images go here
