# Arguments --url (Video URL), --start (Start time in video in seconds), --end (End time in video in seconds),
# --stage (Stage 1 or stage 2, default: 1), --timedelay' (Negative time (s) between video start and T0, default: 0.0)
# and --plottype (Plot or scatter data, default: no plot)
#
# Example: python falcon_data.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 1193 --end 1724 --stage 1
# --timedelay 0.0 --plottype scatter
import math
import re
import cv2
import pafy                         # install with: pip install git+https://github.com/Cupcakus/pafy
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

    url = arguments.url
    video_start_time = arguments.start
    video_end_time = arguments.end
    stage = arguments.stage
    tf = arguments.timedelay

    df = pd.DataFrame(columns=["t", "v", "h"])

    milliseconds = 1000
    frame_number = 0
    fps = 30

    video = pafy.new(url)

    stream_720mp4 = None

    for stream in video.allstreams:
        if stream.resolution == "1280x720" and stream.extension == "mp4":
            stream_720mp4 = stream

    if stream_720mp4 is None:
        print("No 720p mp4 stream found")
        quit()

    csv_filename = "".join(x for x in video.title if x.isalnum()) + "_stage" + stage + ".csv"

    cap = cv2.VideoCapture(stream_720mp4.url)

    cap.set(cv2.CAP_PROP_POS_MSEC, video_start_time * milliseconds)

    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= video_end_time * milliseconds:

        frame_number += 1

        t_frame = round(tf, 3)

        ret, frame = cap.read()

        if stage == '1':
            cropped_frame = frame[640:670, 68:264]  # Stage 1
        elif stage == '2':
            cropped_frame = frame[640:670, 1016:1207]  # Stage 2
        else:
            break

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        custom_config = '-l eng --oem 3 --psm 6 '
        text = pytesseract.image_to_string(gray, config=custom_config)

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

    print("Finished! Average fps: " + str(round((frame_number / (time.time() - start_time)), 1)))

    if args.plottype is not None:
        plot_falcon_data(df, arguments)


def plot_falcon_data(df, arguments):
    df = df.dropna()  # Delete rows without values

    # df = df[(np.abs(stats.zscore(df)) < 1.8).all(axis=1)]     # Delete outliers in all columns
    df = df[(np.abs(stats.zscore(df.v)) < 1.8)]  # Delete velocity outliers
    df = df[(np.abs(stats.zscore(df.h)) < 1.8)]  # Delete altitude outliers

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

    parser.add_argument('--url', nargs='?', type=str, help='Video URL')
    parser.add_argument('--start', nargs='?', type=float, help='Start time in video (seconds)')
    parser.add_argument('--end', nargs='?', type=float, help='End time in video (seconds)')
    parser.add_argument('--stage', nargs='?', choices=['1', '2'], help='Stage 1 or stage 2')
    parser.add_argument('--timedelay', nargs='?', type=float, help='Time (s) between video start and T0')
    parser.add_argument('--plottype', nargs='?', choices=['plot', 'scatter'], help='Plot or scatter data')

    args = parser.parse_args()

    if args.url is None:
        print("Pleade add an URL with --url https://youtube.com/abc")
        quit()
    if args.start is None:
        args.start = 0
    if args.end is None:
        print("Pleade add an end time in video in seconds with --end 400")
        quit()
    if args.stage is None:
        args.stage = '1'
    if args.timedelay is None:
        args.timedelay = 0.0

    get_falcon_data(args)
