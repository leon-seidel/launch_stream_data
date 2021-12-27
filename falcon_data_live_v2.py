# Arguments --url (Video URL), --start (Start time in video in seconds), --end (End time in video in seconds), and
# --plot (Plot velocity or altitude, default: velocity)
#
# Example: python falcon_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 1193 --end 1724
# --plot altitude

import os
import re
import cv2
import pafy  # install with: pip install git+https://github.com/Cupcakus/pafy
import time
import argparse
import pytesseract
import numpy as np
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def get_falcon_data(arguments):
    # Plot settings ##########################################################################
    upper_limit_velocity_plot = 30000       # Upper limit of velocity plot
    upper_limit_altitude_plot = 250         # Upper limit of altitude plot
    lower_limit_acceleration_plot = -5      # Lower limit of acceleration plot
    upper_limit_acceleration_plot = 5       # Upper limit of acceleration plot
    # Outlier prevention #####################################################################
    lower_limit_acceleration = -5           # Highest negative acceleration in gs
    upper_limit_acceleration = 5            # Highest positive acceleration in gs
    # General settings #######################################################################
    mean_of_last = 10                       # Mean value of last n acceleration values
    every_n = 15                            # Only analyse every nth frame
    fps = 30                                # Video fps
    # Setup 0 values #########################################################################
    tf = 0                                  # Time between video start and T0
    frame_number = 0                        # Number of frame

    url = arguments.url
    video_start_time = arguments.start
    video_end_time = arguments.end

    video = pafy.new(url)

    t, v, h, a, a_mean = [[], []], [[], []], [[], []], [[], []], [[], []]

    plt.ion()
    fig1, ax1 = plt.subplots()
    sc_velo1 = ax1.scatter(t[0], v[0])
    sc_velo2 = ax1.scatter(t[1], v[1])
    plt.title(video.title + ": Time vs. velocity")
    plt.legend(["Stage 1", "Stage 2"])
    plt.xlim(0, video_end_time - video_start_time)
    plt.ylim(0, upper_limit_velocity_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Velocity in kph")
    plt.grid()
    plt.draw()

    plt.ion()
    fig2, ax2 = plt.subplots()
    sc_alti1 = ax2.scatter(t[0], h[0])
    sc_alti2 = ax2.scatter(t[1], h[1])
    plt.title(video.title + ": Time vs. altitude")
    plt.legend(["Stage 1", "Stage 2"])
    plt.xlim(0, video_end_time - video_start_time)
    plt.ylim(0, upper_limit_altitude_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Altitude in km")
    plt.grid()
    plt.draw()

    plt.ion()
    fig3, ax3 = plt.subplots()
    sc_acc1 = ax3.scatter(t[0], a_mean[0])
    sc_acc2 = ax3.scatter(t[1], a_mean[1])
    plt.title(video.title + ": Time vs. acceleration")
    plt.legend(["Stage 1", "Stage 2"])
    plt.xlim(0, video_end_time - video_start_time)
    plt.ylim(lower_limit_acceleration_plot, upper_limit_acceleration_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Acceleration in gs")
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
        tf += (1 / fps)

        ret, frame = cap.read()

        if p != 0:
            continue

        t_frame = round(tf, 3)
        print()

        for stage in range(1, 3):

            if stage == 1:
                cropped_frame = frame[640:670, 68:264]  # Stage 1
            else:
                cropped_frame = frame[640:670, 1016:1207]  # Stage 2

            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            custom_config = '--oem 3 --psm 6 '
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

            try:
                n = False
                m = 0
                while n is False:
                    m += 1
                    if not [x for x in (v[stage - 1][-m], t[stage - 1][-m], v_frame, t_frame) if x is None]:
                        a_frame = ((v_frame - v[stage - 1][-m]) / (3.6 * 9.81)) / (t_frame - t[stage - 1][-m])
                        n = True
                    elif [x for x in (v_frame, t_frame) if x is None]:
                        a_frame = None
                        n = True
            except IndexError:
                a_frame = 0

            if a_frame is not None and lower_limit_acceleration <= a_frame <= upper_limit_acceleration:
                if stage == 2 and v_frame is not None:
                    try:
                        if v_frame < v[0][-1]:
                            continue
                    except IndexError:
                        continue

                t[stage - 1].append(t_frame)
                v[stage - 1].append(v_frame)
                h[stage - 1].append(h_frame)
                a[stage - 1].append(a_frame)

                try:
                    n = 0
                    m = 0
                    n_last = []
                    while n < mean_of_last:
                        m += 1
                        if a[stage - 1][-m] is not None:
                            n_last.append(a[stage - 1][-m])
                            n += 1
                    a_frame_mean = mean(n_last)
                except IndexError:
                    a_frame_mean = None
                except TypeError:
                    a_frame_mean = None

                a_mean[stage - 1].append(a_frame_mean)

                print("Stage", stage, ": t=", t_frame, "s, v=", v_frame, "kph, h=", h_frame, "km, a=", a_frame, "gs")

        time_passed = time.time() - start_time
        average_fps = frame_number / time_passed

        print("Average fps: " + str(round(average_fps, 2)) + ", total time: " + str(round(time_passed, 2)) + " s")

        sc_velo1.set_offsets(np.c_[t[0], v[0]])
        sc_velo2.set_offsets(np.c_[t[1], v[1]])
        fig1.canvas.draw_idle()
        plt.pause(0.001)

        sc_alti1.set_offsets(np.c_[t[0], h[0]])
        sc_alti2.set_offsets(np.c_[t[1], h[1]])
        fig2.canvas.draw_idle()
        plt.pause(0.001)

        sc_acc1.set_offsets(np.c_[t[0], a_mean[0]])
        sc_acc2.set_offsets(np.c_[t[1], a_mean[1]])
        fig3.canvas.draw_idle()
        plt.pause(0.001)

    print("Finished!")
    plt.waitforbuttonpress()

    cap.release()
    cv2.destroyAllWindows()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = 'mission_data'
    csv_filename = os.path.join(file_dir, csv_folder, "".join(x for x in video.title if x.isalnum()) + ".csv")
    df = pd.DataFrame(list(zip(t[0], v[0], h[0], v[1], h[1])), columns=["t", "v1", "h1", "v2", "h2"])
    df.to_csv(csv_filename, index=False)

    print("Saved data!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from SpaceX Falcon 9 starts')

    parser.add_argument('--url', nargs='?', type=str, help='Video URL')
    parser.add_argument('--start', nargs='?', type=float, help='Start time in video (seconds)')
    parser.add_argument('--end', nargs='?', type=float, help='End time in video (seconds)')

    args = parser.parse_args()

    if args.url is None:
        print("Pleade add an URL with --url https://www.youtube.com/watch?v=JBGjE9_aosc")
        quit()
    if args.start is None:
        args.start = 0
    if args.end is None:
        print("Pleade add an end time in video in seconds with --end 400")
        quit()

    get_falcon_data(args)
