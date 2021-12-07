import re
import cv2
import csv
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

###################################################################################################
# Paramteres ######################################################################################
###################################################################################################
cap = cv2.VideoCapture('DART Mission_short.mp4')        # Video filename
fps = 30                                                # Video framerate
stage1 = True                                           # False: Stage 2
csv_filename = 'falcon_data_stage1.csv'                 # Data file for csv
tf = -21.0                                              # Seconds between video start and T0
###################################################################################################

number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 1
start_time = time.time()

while cap.isOpened():

    if frame_number == number_of_frames:
        break

    frame_number += 1

    t_frame = round(tf, 3)

    ret, frame = cap.read()

    if stage1 is True:
        cropped_frame = frame[960:1005, 103:395]  # Stage 1
    else:
        cropped_frame = frame[960:1005, 1525:1820]  # Stage 2

    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    text_list = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", text)

    if len(text_list) == 2:

        try:
            v_frame = float(text_list[0])
        except ValueError:
            v_frame = None

        try:
            h_frame = float(text_list[1])
        except ValueError:
            h_frame = None

    else:
        v_frame = None
        h_frame = None

    tf = tf + (1 / fps)

    print(t_frame, v_frame, h_frame)

    print("--- %s seconds ---\n" % round((time.time() - start_time), 2))

    with open(csv_filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([t_frame, v_frame, h_frame])

cap.release()
cv2.destroyAllWindows()

print("Finished! Average time per frame: " + str(round(((time.time() - start_time) / number_of_frames), 3)) + " s.")
