# Get live data and plots from rocket launch live streams
#
# Arguments: --url (Video URL), --start (Start time in video), --duration (Duration of video from start time), supported
# formats: 1:13:12, 3:12, 144 (h:min:s, min:s, s)). For livestreams just use -- start live and a duration.
#
# Example 1: python rocket_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 19:53 --duration 8:24
# Example 2: python rocket_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start live --duration 8:45

import os
import re
import cv2
import pafy  # Install with: pip install git+https://github.com/Cupcakus/pafy
import time
import argparse
import pytesseract
import numpy as np
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def get_rocket_data(arguments):
    ##################################################################################################################
    # Plot settings ##################################################################################################
    upper_limit_velo_plot = 30000           # Upper limit of velocity plot
    upper_limit_alti_plot = 250             # Upper limit of altitude plot
    lower_limit_acc_plot = -5               # Lower limit of acceleration plot
    upper_limit_acc_plot = 5                # Upper limit of acceleration plot
    # Outlier prevention #############################################################################################
    lower_limit_acc = -5                    # Highest negative acceleration in gs
    upper_limit_acc = 5                     # Highest positive acceleration in gs
    lower_limit_v_vert = -10                # Highest negative vertical velocity in km/s
    upper_limit_v_vert = 10                 # Highest positive vertical velocity in km/s
    mean_of_last = 10                       # Mean value of last n acceleration values
    every_n = 15                            # Only analyse every nth frame
    # Telemetry data sources #########################################################################################
    # contains [y_start, y_end, x_start, x_end] of the bounding box ##################################################
    f9_stage1 = [640, 670, 68, 264]         # Position of telemetry data in 720p video feed (Falcon 9, stage 1)
    f9_stage2 = [640, 670, 1016, 1207]      # Position of telemetry data in 720p video feed (Falcon 9, stage 2)
    rocketlab = [35, 55, 976, 1124]         # Position of telemetry data in 720p video feed (Rocket Lab Electron)
    jwst = [542, 685, 170, 248]             # Position of telemetry data in 720p video feed (JWST stream Arianespace)
    labpadre = [0, 30, 1140, 1205]          # Position of clock in livestream (just for livestream testing)
    ##################################################################################################################

    url = arguments.url
    video_start_time, video_end_time, video_duration, is_live = get_video_times(arguments)

    video = pafy.new(url)
    video_title = video.title
    video_author = video.author

    t, v, h, a, v_vert, a_mean = [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]

    if video_author == "SpaceX":
        pos_stage = [f9_stage1, f9_stage2]
    elif video_author == "Rocket Lab":
        pos_stage = [rocketlab]
    elif video_author == "arianespace":
        pos_stage = [jwst]
    elif video_author == "LabPadre":
        pos_stage = [labpadre]
    else:
        pos_stage = None
        print("Youtube channel " + video_author + " not supported.")
        quit()

    number_of_stages = len(pos_stage)  # Number of rocket stages with data

    fig, ax, sc = start_plots(number_of_stages, video_title, upper_limit_velo_plot, upper_limit_alti_plot,
                              upper_limit_acc_plot, lower_limit_acc_plot, video_duration, t, v, h, a_mean)

    stream_720 = None

    if is_live is True:
        for stream in video.allstreams:
            if stream.resolution == "1280x720" and stream.extension == "mp4":
                stream_720 = stream
    else:
        for stream in video.allstreams:
            if stream.resolution == "1280x720" and stream.extension == "webm":
                stream_720 = stream

    if stream_720 is None:
        print("No adequate 720p stream found")
        quit()

    cap = cv2.VideoCapture(stream_720.url)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if is_live is False:
        cap.set(cv2.CAP_PROP_POS_MSEC, video_start_time * 1000)
        true_video_end_time = video_end_time * 1000
    else:
        true_video_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) + video_duration * 1000

    start_time = time.time()
    frame_number = 0
    tf = 0  # Time between video start and T0

    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= true_video_end_time:

        p = (frame_number / every_n) - (int(frame_number / every_n))
        frame_number += 1
        tf += (1 / fps)

        ret, frame = cap.read()

        if p != 0:
            continue

        t_frame = round(tf, 3)
        print()

        for stage in range(1, number_of_stages + 1):

            v_frame, h_frame = get_text_from_frame(video_author, frame, pos_stage, stage)

            a_frame = calculate_acc(t, v, t_frame, v_frame, stage)

            v_vert_frame = calculate_v_vert(t, h, t_frame, h_frame, stage)

            if (a_frame is not None and lower_limit_acc <= a_frame <= upper_limit_acc and
                    lower_limit_v_vert <= v_vert_frame <= upper_limit_v_vert):
                if stage == 2 and v_frame is not None:
                    try:
                        if v_frame < v[0][-1] or h_frame < h[0][-1]:
                            continue
                    except IndexError:
                        t[stage - 1].append(None)
                        v[stage - 1].append(None)
                        h[stage - 1].append(None)
                        a[stage - 1].append(None)
                        v_vert[stage - 1].append(None)
                        a_mean[stage - 1].append(None)
                        continue
                    except TypeError:
                        t[stage - 1].append(None)
                        v[stage - 1].append(None)
                        h[stage - 1].append(None)
                        a[stage - 1].append(None)
                        v_vert[stage - 1].append(None)
                        a_mean[stage - 1].append(None)
                        continue

                t[stage - 1].append(t_frame)
                v[stage - 1].append(v_frame)
                h[stage - 1].append(h_frame)
                a[stage - 1].append(a_frame)
                v_vert[stage - 1].append(v_vert_frame)

                a_frame_mean = calculate_a_mean(a, stage, mean_of_last)

                a_mean[stage - 1].append(a_frame_mean)

                print("Stage", stage, ": t=", t_frame, "s, v=", v_frame, "kph, h=", h_frame, "km, a=", a_frame, "gs")
            else:
                t[stage - 1].append(None)
                v[stage - 1].append(None)
                h[stage - 1].append(None)
                a[stage - 1].append(None)
                v_vert[stage - 1].append(None)
                a_mean[stage - 1].append(None)

        time_passed = time.time() - start_time
        average_fps = frame_number / time_passed

        print("Average fps: " + str(round(average_fps, 2)) + ", total time: " + str(round(time_passed, 2)) + " s")

        update_plots(number_of_stages, t, v, h, a_mean, fig, sc)

    print("Finished!")
    plt.waitforbuttonpress()

    cap.release()
    cv2.destroyAllWindows()

    save_as_csv(t, v, h, a_mean, number_of_stages, video_title)


def get_video_times(arguments):
    video_start_time, video_end_time, video_duration, is_live = 0, 0, 0, False

    if ":" in arguments.duration:
        duration_list = arguments.duration.split(":")

        if len(duration_list) == 2:
            video_duration = float(duration_list[0]) * 60 + float(duration_list[1])
        if len(duration_list) == 3:
            video_duration = float(duration_list[0]) * 3600 + float(duration_list[1]) * 60 + float(duration_list[2])
    else:
        video_duration = float(arguments.duration)

    if arguments.start == "live":
        is_live = True
        video_end_time = video_duration
        return video_start_time, video_end_time, video_duration, is_live
    else:
        if ":" in arguments.start:
            start_list = arguments.start.split(":")

            if len(start_list) == 2:
                video_start_time = float(start_list[0]) * 60 + float(start_list[1])
            if len(start_list) == 3:
                video_start_time = float(start_list[0]) * 3600 + float(start_list[1]) * 60 + float(start_list[2])
        else:
            video_start_time = float(arguments.start)

        video_end_time = video_start_time + video_duration
        return video_start_time, video_end_time, video_duration, is_live


def start_plots(number_of_stages, video_title, upper_limit_velo_plot, upper_limit_alti_plot, upper_limit_acc_plot,
                lower_limit_acc_plot, video_duration, t, v, h, a_mean):
    fig, ax, sc = [], [], [[], [], []]

    # Velocity plot
    plt.ion()
    fig_velo, ax_velo = plt.subplots()
    fig.append(fig_velo)
    ax.append(ax_velo)

    for stage in range(1, number_of_stages + 1):
        sc[0].append(ax[0].scatter(t[stage - 1], v[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. velocity")
    plt.xlim(0, video_duration)
    plt.ylim(0, upper_limit_velo_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Velocity in kph")
    plt.grid()
    plt.draw()

    # Altitude plot
    plt.ion()
    fig_alti, ax_alti = plt.subplots()
    fig.append(fig_alti)
    ax.append(ax_alti)

    for stage in range(1, number_of_stages + 1):
        sc[1].append(ax[1].scatter(t[stage - 1], h[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. altitude")
    plt.xlim(0, video_duration)
    plt.ylim(0, upper_limit_alti_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Altitude in km")
    plt.grid()
    plt.draw()

    # Acceleration plot
    plt.ion()
    fig_acc, ax_acc = plt.subplots()
    fig.append(fig_acc)
    ax.append(ax_acc)

    for stage in range(1, number_of_stages + 1):
        sc[2].append(ax[2].scatter(t[stage - 1], a_mean[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. acceleration")
    plt.xlim(0, video_duration)
    plt.ylim(lower_limit_acc_plot, upper_limit_acc_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Acceleration in gs")
    plt.grid()
    plt.draw()

    return fig, ax, sc


def get_text_from_frame(video_author, frame, pos_stage, stage):
    cropped = frame[pos_stage[stage - 1][0]:pos_stage[stage - 1][1], pos_stage[stage - 1][2]:pos_stage[stage - 1][3]]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

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

    elif video_author == "arianespace" and len(text_list) == 4:
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

    return v_frame, h_frame


def calculate_acc(t, v, t_frame, v_frame, stage):
    try:
        m = 0
        while True:
            m += 1
            if not [x for x in (v[stage - 1][-m], t[stage - 1][-m], v_frame, t_frame) if x is None]:
                a_frame = ((v_frame - v[stage - 1][-m]) / (3.6 * 9.81)) / (t_frame - t[stage - 1][-m])
                return a_frame
            elif [x for x in (v_frame, t_frame) if x is None]:
                a_frame = None
                return a_frame
    except IndexError:
        a_frame = 0
        return a_frame


def calculate_v_vert(t, h, t_frame, h_frame, stage):
    try:
        m = 0
        while True:
            m += 1
            if not [x for x in (h[stage - 1][-m], t[stage - 1][-m], h_frame, t_frame) if x is None]:
                v_vert_frame = (h_frame - h[stage - 1][-m]) / (t_frame - t[stage - 1][-m])
                return v_vert_frame
            elif [x for x in (h_frame, t_frame) if x is None]:
                v_vert_frame = None
                return v_vert_frame
    except IndexError:
        v_vert_frame = 0
        return v_vert_frame


def calculate_a_mean(a, stage, mean_of_last):
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

    return a_frame_mean


def update_plots(number_of_stages, t, v, h, a_mean, fig, sc):
    for stage in range(1, number_of_stages + 1):
        sc[0][stage - 1].set_offsets(np.c_[t[stage - 1], v[stage - 1]])
    fig[0].canvas.draw_idle()
    plt.pause(0.001)

    for stage in range(1, number_of_stages + 1):
        sc[1][stage - 1].set_offsets(np.c_[t[stage - 1], h[stage - 1]])
    fig[1].canvas.draw_idle()
    plt.pause(0.001)

    for stage in range(1, number_of_stages + 1):
        sc[2][stage - 1].set_offsets(np.c_[t[stage - 1], a_mean[stage - 1]])
    fig[2].canvas.draw_idle()
    plt.pause(0.001)


def save_as_csv(t, v, h, a_mean, number_of_stages, video_title):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = 'mission_data'
    csv_filename = os.path.join(file_dir, csv_folder, "".join(x for x in video_title if x.isalnum()) + ".csv")

    if number_of_stages == 1:
        column_names = ["t", "v", "h", "a"]
        df_list = list(zip(t[0], v[0], h[0], a_mean[0]))
    elif number_of_stages == 2:
        column_names = ["t", "v1", "h1", "a1", "v2", "h2", "a2"]
        df_list = list(zip(t[0], v[0], h[0], a_mean[0], v[1], h[1], a_mean[1]))
    else:
        df_list, column_names = None, None
        print("Writing a csv of more than two stages is not supported.")
        quit()

    df = pd.DataFrame(df_list, columns=column_names)
    df.to_csv(csv_filename, index=False)

    print("Saved data!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read and plot data from SpaceX F9 and Rocket Lab Electron starts')

    parser.add_argument('--url', nargs='?', type=str, help='Video URL')
    parser.add_argument('--start', nargs='?', type=str, help='Video start time, formats: 1:13:12, 3:12, 144, live')
    parser.add_argument('--duration', nargs='?', type=str, help='Video duration, formats: 1:13:12, 3:12, 144')

    args = parser.parse_args()

    if args.url is None:
        print("Pleade add an URL with --url https://www.youtube.com/watch?v=JBGjE9_aosc")
        quit()
    if args.start is None:
        args.start = "0"
    if args.duration is None:
        print("Pleade add a video duration, supported formats: 1:13:12, 3:12, 144")
        quit()

    get_rocket_data(args)
