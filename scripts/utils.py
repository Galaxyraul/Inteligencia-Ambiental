import numpy as np
import cv2

#Tuplas de tamaño 4 como objetos en el orden xmin, ymin, xmax, ymax
def check_in(outer,points):
    results = []
    outer_xmin, outer_ymin, outer_xmax, outer_ymax = outer
    for point in points:
        inner_xmin, inner_ymin, inner_xmax, inner_ymax = point
        center_x,center_y = (inner_xmax + inner_xmin)/2,(inner_ymax+inner_ymin)/2
        results.append[outer_xmax < center_x < outer_xmin and outer_ymax < center_y < outer_ymin]
    return results

def draw_box(frame,color,classname,conf,xmin,ymin,xmax,ymax):
    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

    label = f'{classname}: {int(conf*100)}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
