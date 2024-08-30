# ImageProcessingLi
An image processing project for lithium metal/ NMC particles - streamline the image processing/ data analysis for R&D labs and university labs.


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

![Histogram of unquantized image](https://github.com/user-attachments/assets/8d132be6-b5d9-4730-915d-6b2c86c97fc4)

![Histogram of quantized image](https://github.com/user-attachments/assets/0a573578-8628-4658-84ab-19dc3a131463)


As you can see, it will be quite difficult to automate the thresholding selection process (which will ultimately become an input in the contouring operation) with an unquantized image. Here are the final results of the area that are chosen to be analyzed - for otsu thresholding of unquantized image and for the thresholding method that we developed for quantized image:

![Portion of image analyzed if Otsu threshold was used on unquantized image](https://github.com/user-attachments/assets/8bb81ede-9d15-4068-be22-bc632ef0d7f5)

![Analyzed image after quantization](https://github.com/user-attachments/assets/95accf0a-3dd5-4238-8f90-5498cdb9fba2)

For the quantization process. To choose the optimal amount of clusters, we used silhouette analysis. I prefer this because it's easier to quantify which are the best choices compared to the somewhat qualitative method of elbow analysis. The silhouette analysis used in this project is based off of the simplified silhouette defined in https://en.wikipedia.org/wiki/Silhouette_(clustering). Note that domain context will still be needed here to determine if the chosen k-clusters make sense. In this case, 3 was chosen because there was the porous section, cathode active material and Al foil chosen. 


## Final results

![Histogram of particle sizes](https://github.com/user-attachments/assets/d0445ef1-a7e8-4cc2-9cdd-3ff0934c86f7)

## Citations:

“JEOL USA Blog: Achieving Pristine Cross Sections of Battery Samples for Scanning Electron Microscopy.” JEOL USA, Inc. Accessed August 30, 2024. https://www.jeolusa.com/NEWS-EVENTS/Blog/pristine-cross-sections-of-batteries. 
“Silhouette (Clustering).” Wikipedia, July 1, 2024. https://en.wikipedia.org/wiki/Silhouette_(clustering). 
“Selecting the Number of Clusters with Silhouette Analysis on Kmeans Clustering.” scikit. Accessed August 30, 2024. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html. 

