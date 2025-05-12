import numpy as np
import cv2

#Tuplas de tamaño 4 como objetos en el orden xmin, ymin, xmax, ymax
def check_in(outer,points):
    results = []
    outer_xmin, outer_ymin, outer_xmax, outer_ymax = outer
    for point in points:
        inner_xmin, inner_ymin, inner_xmax, inner_ymax = point
        center_x,center_y = (inner_xmax + inner_xmin)/2,(inner_ymax+inner_ymin)/2
        results.append(outer_xmax > center_x > outer_xmin and outer_ymax > center_y > outer_ymin)
    return results

def draw_box(frame,color,classname,conf,xmin,ymin,xmax,ymax):
    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

    label = f'{classname}: {int(conf*100)}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

def send_message(client,topic,msg):
    print(f'Sending {msg} to {topic}')
    client.publish(f'parking{topic}',msg)

def get_detections(cap,model):
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        return
    # Resize frame to desired display resolution
    
    frame = cv2.resize(frame,(1080,720))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    return frame,results[0].boxes

def process_detections(frame,detections,labels,min_thresh,states,bbox_colors):
    spots = []
    objects = []
    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:

            if classname in states.keys():
                spots.append([classname,[xmin, ymin, xmax, ymax]])
            else:
                objects.append([xmin, ymin, xmax, ymax])

            draw_box(frame,bbox_colors[classidx % 10],classname,conf,xmin,ymin,xmax,ymax)
    return spots,objects

def display_detections(frame,obj_count):
    # Display detection results
    cv2.putText(frame, f'Number of objects: {obj_count}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO detection results',frame) # Display image

def process_top(spots,objects,states,client):
    for name,box in spots:
        is_in = check_in(box,objects) 
        occupied = any(is_in)
        if occupied and states[name]:
            send_message(client,f'/plaza/{name[-1]}',1)
            states[name] = False

        if not occupied and not states[name]:
            send_message(client,f'/plaza/{name[-1]}',0)
            states[name] = True
                        
            
def get_controls(frame):
    key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        return True
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    return False