import numpy as np

def minimum_selection(rois, amount_selected):

    areas = np.array([])

    for i in range(0, len(rois)):
        # Save the pos of the element and calc the area of the roi
        areas = np.append(areas, [rois[i, 2] * rois[i, 3]], 0)

    data_type = np.uint64
    minimum_rois = np.array([[0, 0, 0, 0]], dtype = data_type)

    for i in range(0, amount_selected):
        # Get the smaller area
        minimum = np.argmin(areas)
        # Added to the selected rois
        minimum_rois = np.append(minimum_rois, [rois[minimum]], 0)
        # Delete that element from the list 
        areas[minimum] = np.iinfo(data_type).max



    return minimum_rois
