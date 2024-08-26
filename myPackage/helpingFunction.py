import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.spatial import distance 


def show_image(my_image, 
               title = None):
    if (my_image.ndim>2):
        my_image = my_image[:,:,::-1] #OpenCV follows BGR, matplotlib likely follows RGB

    fig = plt.subplot(title = str(title))

    fig.imshow(my_image, cmap = 'gray', interpolation = 'bicubic')

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

def kmeans_color_quantization(image, clusters, rounds = 1):

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

def crop_image_white_background(image):
    working_image = image.copy()

    threshold = 250 #Arbitrarily high threshold to catch white background
    assign_value = 255
    threshold_method = cv2.THRESH_BINARY_INV

    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    _,thresholded_image = cv2.threshold(gray, threshold, assign_value, threshold_method)

    #Get external contours
    contours_edge_white_background = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours_edge_white_background)
    contours_edge_white_background = contours_edge_white_background[0] if len(contours_edge_white_background) ==2 else contours_edge_white_background[1]

    for cntr in contours_edge_white_background:
        x,y,w,h = cv2.boundingRect(cntr)
        ROI = working_image[y:y+h, x:x+w]
        break

    show_image(ROI)
    return ROI

def detect_SEM_scale_information():
    
    return None

def simplified_silhouhette_analysis():


    return None 

def update_dict(d,updates):
    d.update(updates)
    return d


def euclidean_calculation(v1,v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


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

def silhouette_analysis_of_kmeans_image_clustering(image_to_be_clustered, range_clusters, silhouette_tol = 0.7, rounds = 1):

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