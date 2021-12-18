# Arguments --filename (Video filename), --fps (Video frames per second, default: 30), --stage (Stage 1 or stage 2,
# default: 1) and --timedelay' (Time (s) between video start and T0, default:0)

import re
import cv2
import csv
import time
import argparse
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
start_time = time.time()


def get_falcon_data(arguments):

    video_filename = arguments.filename
    fps = arguments.fps
    stage = arguments.stage
    tf = arguments.timedelay

    if stage == '1':
        csv_filename = video_filename.split('.')[0] + '_stage1.csv'  # Stage 1
    else:
        csv_filename = video_filename.split('.')[0] + '_stage2.csv'  # Stage 2

    f = open(csv_filename, "w+")
    f.close()

    cap = cv2.VideoCapture(video_filename)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 1

    while cap.isOpened():

        if frame_number == number_of_frames:
            break

        frame_number += 1

        t_frame = round(tf, 3)

        ret, frame = cap.read()

        if stage == '1':
            cropped_frame = frame[960:1005, 103:395]    # Stage 1
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from SpaceX Falcon 9 starts')

    parser.add_argument('--filename', nargs='?', type=str, help='Video filename')
    parser.add_argument('--fps', nargs='?', type=float, help='Video frames per second')
    parser.add_argument('--stage', nargs='?', choices=['1', '2'], help='Stage 1 or stage 2')
    parser.add_argument('--timedelay', nargs='?', type=float, help='Time (s) between video start and T0')

    args = parser.parse_args()

    if args.fps is None:
        args.fps = 30
    if args.stage is None:
        args.stage = '1'
    if args.timedelay is None:
        args.timedelay = 0.0

    get_falcon_data(args)
