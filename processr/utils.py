import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {1: "Class_1", 2: "Class_2", 3: "Class_3"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "area": d.area,
            "perimeter": d.perimeter,
            "compactness": d.compactness,
            "kernel_length": d.kernel_length,
            "kernel_width": d.kernel_width,
            "asymmetry_coefficient": d.asymmetry_coefficient,
            "kernel_groove_length": d.kernel_groove_length,
            "seed_class": d.seed_class,
        }
        for d in data
    ]

    return processed
