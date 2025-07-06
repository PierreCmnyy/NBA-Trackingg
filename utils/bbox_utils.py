def get_center_of_bbox(bbox):
    """Calculates the center of a bounding box."""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def get_bbox_width(bbox) :
    """Calculates the width of a bounding box."""
    #x1, y1, x2, y2 = bbox
    return bbox[2] - bbox[0]