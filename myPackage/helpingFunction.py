import matplotlib.pyplot as plt


def showImage(myImage):
    if (myImage.ndim>2):
        myImage = myImage[:,:,::-1] #OpenCV follows BGR, matplotlib likely follows RGB

    fig = plt.subplot()

    fig.imshow(myImage, cmap = 'gray', interpolation = 'bicubic')

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
             