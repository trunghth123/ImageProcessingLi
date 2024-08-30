import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

from scipy.spatial import distance 
from typing import Optional

def show_image(my_image: np.ndarray , 
               title: Optional[str] = None):
    if (my_image.ndim>2):
        my_image = my_image[:,:,::-1] #OpenCV follows BGR, matplotlib likely follows RGB

    fig = plt.subplot(title = str(title))

    fig.imshow(my_image)

    plt.show()
    return fig

def listNeighbors(x,y, points):
    candidates = [(x-1,y), (x+1,y), (x,y-1), (x,y-1), (x,y+1),
                  (x-1, y-1), (x+1,y+1), (x+1,y-1), (x-1, y+1)]
    return (c for c in candidates if c in points)

class curveAnalyzer:
    def _init_(self, coordinates):
        self.coordinates = sorted(coordinates)
        self.max_x = None
        self.max_y = None
        self.min_x = None
        self.min_y = None
        self.center_of_mass = None
        self._analyze_coordinates()

    def _analyze_coordinates(self):
        if not self.coordinates:
            return
        
        #Initialize max and min with first coordinates
        self.min_x = self.coordinates[0][0]
        self.min_y = self.coordinates[0][1]
        self.max_x = self.coordinates[-1][0]
        self.max_y = self.coordinates[-1][1]
        self.center_of_mass = [(self.min_x+self.max_x)/2, (self.min_y + self.max_y)/2]

    def _get_bounds(self):
        return {
            "max_x": self.max_x,
            "max_y": self.max_y,
            "min_x": self.min_x,
            "min_y": self.min_y 

        }
    
    def _get_center_of_mass(self):
        return { "Center of mass": self.center_of_mass}
    

class curveManager:
    #Manage contours
    def _init_(self):
        self.curves = {}

    def _add_curve(self, name, coordinates):
        if name in self.curves:
            raise ValueError(f"A curve with the name '{name}' already exists.")
        self.curves = {}

    def _get_curve(self, name):
        return self.curves[name]
    
    def _remove_curve(self, name):
        if name in self.curves:
            del self.curves[name]
        else:
            raise ValueError(f"No curve found with the name '{name}")
    
    def _list_curves(self):
        return list(self.curves.keys())
    
    def _print_curves_coordinates(self, index):
        return list(self.curves.values[index])
    
    def _find_curves(self, min_x = None, min_y = None, max_x = None, max_y = None):
        result = []

        for curve in self.curves.values():
            array = list(curve._get_bounds().values())
            c_max_x, c_max_y, c_min_x, c_min_y = array[0], array[1], array[2], array[3]

            #Error handling
            if (c_min_x is None) or (c_min_y is None):
                continue
            if ((min_x is None or c_min_x>= min_x) and 
                (max_x is None or c_max_x <= max_x) and
                (min_y is None or c_min_y >= min_y) and 
                (max_y is None or c_max_y <= max_y)):
                result.append(curve)
        return result
    
    def _find_precise_curve(self, min_x = None, min_y = None, max_x = None, max_y = None, range_x = None, range_y = None):
        result = []

        for curve in self.curves.values():
            array = list(curve.get_bounds().values())
            c_max_x, c_max_y, c_min_x, c_min_y = array[0], array[1], array[2], array[3]

            if (c_min_x is None) or (c_min_y is None):
                continue
            if ((min_x is None or c_min_x == min_x) and 
                (max_x is None or c_max_x == max_x) and 
                (min_y is None or c_min_y >= min_y) and 
                (max_y is None or c_max_y <= max_y)):
                result.append(curve)
            return result
        
    def _get_area_between_two_curves(self, name1, name2):
        curve1 = self.curves[name1]
        curve2 = self.curves[name2]

def pop_all(l):
    r, l[:] = l[:], []
    return r

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def create_samples(image):

    h,w = image.shape[:2]
    samples = np.zeros([h*w, 3], dtype = np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count+=1

    return samples

def kmeans_color_quantization(image: np.ndarray,
                              clusters: int, 
                              rounds: int = 1):

    samples = create_samples(image)

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10**4, 10**(-4)),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape)), compactness, centers, labels

def crop_image_white_background(image: np.ndarray, show_bool = True):

    #SEM images shouldn't have white portion, eliminating cases where screens
    working_image = image.copy()
    h,w = working_image.shape[:2]

    threshold = 250 #Arbitrarily high threshold to catch white background
    assign_value = 255
    threshold_method = cv2.THRESH_BINARY_INV

    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    _,thresholded_image = cv2.threshold(gray, threshold, assign_value, threshold_method)

    #Smoothen the image:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    construct_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (math.floor(h/3),math.floor(w/3)))
    detect_kernel = cv2.morphologyEx(close, cv2.MORPH_OPEN, construct_kernel, iterations=2)
    
    #Get external contours
    contours_edge_white_background, hierarchy = cv2.findContours(detect_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_edge_white_background) != 0:
        biggest_contour = max(contours_edge_white_background,key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(biggest_contour)
        ROI = working_image[y:y+h, x:x+w]
    else:
        print("Unable to find continuous white background")
        ROI = working_image
    if show_bool:
        show_image(ROI, "Region of Interest w/o white background")
        
    return ROI

def detect_SEM_scale_information(image: np.ndarray, height_cropped: int = -100, show_image_bool: bool = True):
    image_cropped = image[height_cropped:, :]
    scale_length = input("What is the length scale of this SEM image?")
    #Text detection and extraction - the bar is usually white on a black font for Jeol images
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY) #Arbitrarily low to catch black section of JEOL images

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
    # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
    # Filtering contours by aspect ratio and size
        aspect_ratio = float(w) / h
        if (aspect_ratio > 10) & (h>1):  # Assuming the scale bar is much wider than its height, arbitrary number & guard against un
        # Draw a rectangle around the detected scale bar (optional visualization)
            new_image = cv2.rectangle(image_cropped, (x, y), (x + w, y + h), (0, 20, 255), 2)
            print(f"x,y,w,h are {x,y,w,h}")
            if show_image_bool:
                show_image(new_image, 'Visualization image')

            print(f"Scale Bar Detected: Width={w} pixels, Height={h} pixels")

            actual_scale_length = scale_length  # nm (replace this with the actual scale value)
            scale_length_per_pixel = float(actual_scale_length) / float(w)
            area_per_pixel_area = scale_length_per_pixel**2
            print(f"Scale length per pixel: {scale_length_per_pixel} um/pixel")
            print(f"Area per area pixel: {area_per_pixel_area} um^2/pixel^2" )
        else:
            continue
    return scale_length_per_pixel, area_per_pixel_area

def update_dict(d: dict, updates: dict):
    d.update(updates)
    return d


def euclidean_calculation(v1,v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def nth_largest(arr, n): 
    return np.partition(arr, -n)[-n] 

def closest_point(point, points): #For adjacent cluster
    if point in points:
        if type(point) is list:
            closest_index = distance.cdist([point], points.remove(point)).argmin()
            closest_point = points.remove(point)[closest_index]
        elif type(point) is np.ndarray:
            closest_index = distance.cdist([point], points[~np.all(points == point,axis = 1)]).argmin()
            closest_point = points[~np.all(points == point,axis = 1)][closest_index]
        
    else:
        closest_index = distance.cdist([point], points).argmin()
        closest_point = points[closest_index]
    return closest_point, closest_index

def elbow_analysis(image_to_be_clustered: np.ndarray, 
                    range_clusters: list, 
                    silhouette_tol: float = 0.7, 
                    rounds: int = 1):
    compact_array = []

    for k in range_clusters:
        quantized_image, compactness, centers, labels = kmeans_color_quantization(image_to_be_clustered, clusters=k, rounds=rounds)
        compact_array.append(1/k*compactness)
    plt.scatter(range_clusters, compact_array)
    plt.show()

    return compact_array


def silhouette_analysis_of_kmeans_image_clustering(image_to_be_clustered: np.ndarray, 
                                                   range_clusters: list, 
                                                   silhouette_tol: float = 0.7, 
                                                   rounds: int = 1):
    #Get a graph silhouette coefficient vs cluster numbers to determine the most optimal clusters to choose

    h,w = image_to_be_clustered.shape[:2]

    #Initialize dictionary of number of clusters and their respective silhouette score
    cluster_dictionary = {}
    for i in range_clusters:
        cluster_dictionary[i] = None

    for clusters_number in range_clusters:
        s_prime_i_array = []
        samples = create_samples(image_to_be_clustered)
        quantized_image, compactness, centers, labels = kmeans_color_quantization(image_to_be_clustered, clusters=clusters_number, rounds=rounds)
        quantized_image_array = centers[labels.flatten()]

        for center in centers:
            matching_indices = np.where(np.all(quantized_image_array == center, axis=1))[0]
            a_prime_i_center = distance.cdist([center], samples[matching_indices], 'euclidean').sum()
            closest_center, closest_index = closest_point(center, centers)
            b_prime_i_center0 = distance.cdist([closest_center], samples[matching_indices], 'euclidean').sum()
            s_prime_i = 1 - a_prime_i_center/b_prime_i_center0
            s_prime_i_array.append(s_prime_i)
        
        silhouette_coefficient_dict = {clusters_number: sum(s_prime_i_array)/len(centers)}
        cluster_dictionary = update_dict(cluster_dictionary, silhouette_coefficient_dict)
    
    #Show results:
    keys = list(cluster_dictionary.keys())
    values = list(cluster_dictionary.values())

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(keys, values, marker='o', linestyle='-', color='b')

    plt.axhline(y=silhouette_tol, color='r', linestyle='--', label='y = 0.7')

    # Add titles and labels
    plt.title('Silhouette coefficient plot')
    plt.xlabel('Clusters')
    plt.ylabel('Values')

    #Set limit
    plt.xlim(min(range_clusters), max(range_clusters))
    plt.ylim(0, 1)

    # Show the plot
    plt.grid(True)
    plt.show()

    return cluster_dictionary


def plot_top_two_quantized_images(image_to_be_clustered, cluster_dictionary, rounds=1):

    sorted_clusters = sorted(cluster_dictionary.items(), key=lambda item: item[1], reverse=True)
    top_two_clusters = [sorted_clusters[0][0], sorted_clusters[1][0]]  # Extracting the keys (cluster numbers) of the two highest values

    for clusters_number in top_two_clusters:
        quantized_image, _, centers, _ = kmeans_color_quantization(image_to_be_clustered, clusters=clusters_number, rounds=rounds)
        
        # Plot the quantized image
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct color display
        plt.title(f'Quantized Image with {clusters_number} Clusters')
        plt.axis('off')
        plt.show()

    return top_two_clusters


def show_histogram(image):
    #Calculate histogram of this image

    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    normalizeHist = hist/hist.sum() 


    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(hist)
    plt.xlim([0,256])

    return hist

def show_size_histogram(size_list):
    # Check if the list is empty
    if not size_list:
        print("The size list is empty. No histogram to display.")
        return

    # Create a histogram
    plt.hist(size_list, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Size List')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def remove_outliers(data):
    data_array = np.array(data)
    
    # Calculate the Interquartile Range (IQR)
    Q1 = np.percentile(data_array, 25)
    Q3 = np.percentile(data_array, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data to exclude outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data


def find_particle_size(image, histogram, per_pixel_area):

    #This function will find the interquartile size distribution of all the particles
    #Preprocessing
    processed_image = crop_image_white_background(image)
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    #Initialization
    points_list = []
    size_list = []
    assign_value = 255
    threshold_method = cv2.THRESH_BINARY

    #Assumption 1: the image is primarily filled with articles need analyzing, gap is darker
    #Assumption 2: Atomic number of Aluminum is lower than that of NMC - thus appearing darker in the final image
    #Assumption 2 should be reversed while analyzing the anode since Cu > C/ Li-anode

    max_histogram_peak_location = int(np.where(histogram == np.unique(histogram)[-1])[0]) #Histogram is a sorted array from 0-256 containing number of pixels
    second_largest_histogram_peak = int(np.where(histogram == np.unique(histogram)[-2])[0])
    threshold_value = (max_histogram_peak_location+second_largest_histogram_peak)/2 #In between point
    print(threshold_value)

    _,result = cv2.threshold(gray_image, threshold_value, assign_value, threshold_method)
   
    show_image(result, 'Image after thresholding')

    cnts, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    AREA_THRESHOLD = 1

    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            cv2.drawContours(result, [c], -1, 0, -1)
        else:
            (x, y), radius = cv2.minEnclosingCircle(c)
            points_list.append((int(x), int(y)))
            size_list.append(area)

    #Highlight analyzed section w/ green
    final_result = cv2.bitwise_and(processed_image, processed_image, mask=result)
    final_result[result==255] = (36,255,12)

    
    filtered_size = np.array(remove_outliers(size_list))*per_pixel_area #Get the interquartile 
    show_size_histogram(list(filtered_size))

    return size_list
