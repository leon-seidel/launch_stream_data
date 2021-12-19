# Arguments --filename (Video filename), --fps (Video frames per second, default: 30), --stage (Stage 1 or stage 2,
# default: 1), --timedelay' (Negative time (s) between video start and T0, default:0) and --plottype (Plot or scatter
# data, default: no plot)

import re
import cv2
import time
import argparse
import pytesseract
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
start_time = time.time()


def get_falcon_data(arguments):

    video_filename = arguments.filename
    fps = arguments.fps
    stage = arguments.stage
    tf = arguments.timedelay

    df = pd.DataFrame(columns=["t", "v", "h"])

    if stage == '1':
        csv_filename = video_filename.split('.')[0] + '_stage1.csv'  # Stage 1
    elif stage == '2':
        csv_filename = video_filename.split('.')[0] + '_stage2.csv'  # Stage 2
    else:
        quit()

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
        elif stage == '2':
            cropped_frame = frame[960:1005, 1525:1820]  # Stage 2
        else:
            break

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

        df = df.append({'t': t_frame, 'v': v_frame, 'h': h_frame}, ignore_index=True)

    cap.release()
    cv2.destroyAllWindows()

    df.to_csv(csv_filename, index=False)

    print("Finished! Average time per frame: " + str(round(((time.time() - start_time) / number_of_frames), 3)) + " s.")

    if args.plottype is not None:
        plot_falcon_data(df, arguments)


def plot_falcon_data(df, arguments):

    df = df.dropna()                                            # Delete rows without values

    # df = df[(np.abs(stats.zscore(df)) < 1.8).all(axis=1)]     # Delete outliers in all columns
    df = df[(np.abs(stats.zscore(df.v)) < 1.8)]                 # Delete velocity outliers
    df = df[(np.abs(stats.zscore(df.h)) < 1.8)]                 # Delete altitude outliers

    if arguments.plottype == "plot":
        plt.plot(df.t, df.v)
    elif arguments.plottype == "scatter":
        plt.scatter(df.t, df.v)

    plt.title("Time vs. velocity of stage " + arguments.stage)
    plt.xlabel("Time in s")
    plt.ylabel("Velocity in kph")
    plt.grid()
    plt.show()

    if arguments.plottype == "plot":
        plt.plot(df.t, df.h)
    elif arguments.plottype == "scatter":
        plt.scatter(df.t, df.h)

    plt.title("Time vs. altitude of stage " + arguments.stage)
    plt.xlabel("Time in s")
    plt.ylabel("Altitude in km")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from SpaceX Falcon 9 starts')

    parser.add_argument('--filename', nargs='?', type=str, help='Video filename')
    parser.add_argument('--fps', nargs='?', type=float, help='Video frames per second')
    parser.add_argument('--stage', nargs='?', choices=['1', '2'], help='Stage 1 or stage 2')
    parser.add_argument('--timedelay', nargs='?', type=float, help='Time (s) between video start and T0')
    parser.add_argument('--plottype', nargs='?', choices=['plot', 'scatter'], help='Plot or scatter data')

    args = parser.parse_args()

    if args.filename is None:
        print("Pleade add a filename with --filename abc.mp4")
        quit()
    if args.fps is None:
        args.fps = 30
    if args.stage is None:
        args.stage = '1'
    if args.timedelay is None:
        args.timedelay = 0.0

    get_falcon_data(args)
