import paddle
from paddleocr import PaddleOCR
import cv2
from utils import check_in,send_message

paddle.set_device('cpu') 
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 

def process_front(frame,spots,objects,states,client):
    for name, box in spots:
        is_in = check_in(box, objects)
        to_read = any(is_in)
        if to_read:
            if states[name]:  
                # Find the first object that is inside
                for idx, inside in enumerate(is_in):
                    if inside:
                        obj_box = objects[idx]
                        xmin, ymin, xmax, ymax = map(int, obj_box)

                        # Crop the object
                        crop = frame[ymin:ymax, xmin:xmax]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        result = ocr.ocr(crop_rgb, cls=True)

                        # Process OCR results and print detected text
                        if result:
                            for line in result:
                                if line:
                                    for word_info in line:
                                        text = word_info[1][0]          # Detected text
                                        confidence = word_info[1][1]    # Confidence score
                                        if confidence > 0.5:  # Adjust this threshold as needed
                                            send_message(client,f'/matricula/{name[-1]}',text)
                                            print(f"Detected Text: {text} with confidence: {confidence:.2f}")
                                            states[name] = False
                            break 