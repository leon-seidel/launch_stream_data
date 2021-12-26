# Arguments --url (Video URL), --start (Start time in video in seconds), --end (End time in video in seconds), and
# --plot (Plot velocity or altitude, default: velocity)
#
# Example: python falcon_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 1193 --end 1724
# --plot altitude

import os
import re
import cv2
import pafy                         # install with: pip install git+https://github.com/Cupcakus/pafy
import time
import argparse
import pytesseract
import numpy as np
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def get_falcon_data(arguments):

    upper_limit_velocity_plot = 36000
    upper_limit_altitude_plot = 1500
    lower_limit_acceleration_plot = -1
    upper_limit_acceleration_plot = 4
    mean_of_last = 30                       # Mean value of last n acceleration values
    every_n = 15                            # Only analyse every nth frame
    fps = 30                                # Video fps

    tf = 60
    frame_number = 0

    url = arguments.url
    video_start_time = arguments.start
    video_end_time = arguments.end

    video = pafy.new(url)
    t, v, h, a, a_mean = [], [], [], [], []

    plt.ion()
    fig1, ax1 = plt.subplots()
    sc_velo = ax1.scatter(t, v)
    plt.ylim(0, upper_limit_velocity_plot)
    plt.title("JWST launch: Time vs. velocity")
    plt.ylabel("Velocity in kph")
    plt.xlim(0, video_end_time - video_start_time)
    plt.xlabel("Time in s")
    plt.grid()
    plt.draw()

    plt.ion()
    fig2, ax2 = plt.subplots()
    sc_alti = ax2.scatter(t, h)
    plt.ylim(0, upper_limit_altitude_plot)
    plt.title("JWST launch: Time vs. altitude")
    plt.ylabel("Altitude in km")
    plt.xlim(0, video_end_time - video_start_time)
    plt.xlabel("Time in s")
    plt.grid()
    plt.draw()

    plt.ion()
    fig3, ax3 = plt.subplots()
    sc_acc = ax3.scatter(t, h)
    plt.ylim(lower_limit_acceleration_plot, upper_limit_acceleration_plot)
    plt.title("JWST launch: Time vs. acceleration")
    plt.ylabel("Acceleration in gs")
    plt.xlim(0, video_end_time - video_start_time)
    plt.xlabel("Time in s")
    plt.grid()
    plt.draw()

    stream_720mp4 = None

    for stream in video.allstreams:
        if stream.resolution == "1280x720" and stream.extension == "webm":
            stream_720mp4 = stream

    if stream_720mp4 is None:
        print("No 720p mp4 stream found")
        quit()

    cap = cv2.VideoCapture(stream_720mp4.url)

    cap.set(cv2.CAP_PROP_POS_MSEC, video_start_time * 1000)

    start_time = time.time()

    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= video_end_time * 1000:

        p = (frame_number / every_n) - (int(frame_number / every_n))
        frame_number += 1
        t_frame = round(tf, 3)

        tf += (1 / fps)

        ret, frame = cap.read()

        if p != 0:
            continue

        print()

        cropped_frame = frame[542:685, 170:248]

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        custom_config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, lang="eng", config=custom_config)

        text_list = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", text)

        if len(text_list) == 4:

            try:
                v_frame = round((float(text_list[2]) + (float(text_list[3]) / 100)) * 3600, 1)
            except ValueError:
                v_frame = None

            try:
                h_frame = float(text_list[0])
            except ValueError:
                h_frame = None

        else:
            v_frame = None
            h_frame = None

        try:
            n = False
            m = 0
            while n is False:
                m += 1
                if v[-m] is not None and t[-m] is not None and v_frame is not None and t_frame is not None:
                    a_frame = ((v_frame - v[-m]) / (3.6 * 9.81)) / (t_frame - t[-m])
                    n = True
                elif v[-m] is not None and t[-m] is not None:
                    a_frame = None
                    n = True
                else:
                    a_frame = None
        except IndexError:
            a_frame = None

        t.append(t_frame)
        v.append(v_frame)
        h.append(h_frame)
        a.append(a_frame)

        try:
            n = 0
            m = 0
            n_last = []
            while n < mean_of_last:
                m += 1
                if a[-m] is not None:
                    n_last.append(a[-m])
                    n += 1
            a_frame_mean = mean(n_last)
        except IndexError:
            a_frame_mean = None
        except TypeError:
            a_frame_mean = None

        a_mean.append(a_frame_mean)

        print("t=", t_frame, "s, v=", v_frame, "kph, h=", h_frame, "km, a_mean=", a_frame_mean, "gs, a=", a_frame)

        time_passed = time.time() - start_time
        average_fps = frame_number / time_passed

        print("Average fps: " + str(round(average_fps, 2)) + ", total time: " + str(round(time_passed, 2)) + " s")

        sc_velo.set_offsets(np.c_[t, v])
        fig1.canvas.draw_idle()
        plt.pause(0.0001)

        sc_alti.set_offsets(np.c_[t, h])
        fig2.canvas.draw_idle()
        plt.pause(0.0001)

        sc_acc.set_offsets(np.c_[t, a_mean])
        fig3.canvas.draw_idle()
        plt.pause(0.001)

    plt.waitforbuttonpress()

    cap.release()
    cv2.destroyAllWindows()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = 'mission_data'
    csv_filename = os.path.join(file_dir, csv_folder, "JamesWebbSpaceTelescopeLaunch.csv")
    df = pd.DataFrame(list(zip(t, v, h)), columns=["t", "v", "h"])
    df.to_csv(csv_filename, index=False)

    print("Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from JWST start')

    parser.add_argument('--url', nargs='?', type=str, help='Video URL')
    parser.add_argument('--start', nargs='?', type=float, help='Start time in video (seconds)')
    parser.add_argument('--end', nargs='?', type=float, help='End time in video (seconds)')
    parser.add_argument('--plot', nargs='?', choices=['velocity', 'altitude'], help='Plot velocity or altitude')

    args = parser.parse_args()

    if args.url is None:
        print("Pleade add an URL with --url https://www.youtube.com/watch?v=JBGjE9_aosc")
        quit()
    if args.start is None:
        args.start = 0
    if args.end is None:
        print("Pleade add an end time in video in seconds with --end 400")
        quit()
    if args.plot is None:
        args.plot = "velocity"

    get_falcon_data(args)
