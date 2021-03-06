# Plot rocket launch data from YouTube streams

## General
![Acceleration plot](examples/transporter3_acc.png?raw=true)

Using PyTesseract for extracting the telemetry data from SpaceX, Rocket Lab and Arianespace launch streams.
Velocity, altitude and acceleration are then plotted for each stage (SpaceX) or for the main stage (Rocket Lab, Arianespace).
The rocket type is determined by the YouTube channel name, supporting the above rocket launchers. Outliers are detected and ignored by applying acceleration and vertical speed boundaries.
Realtime performance can be reached by only analysing every nth frame. Accelerations are the combination of velocity change
rates and acceleration due to gravitational forces. Furthermore, accelerations are shown as a moving average.

## Arguments
Arguments: `--url` (Video URL), `--start` (Start time in video in seconds), `--duration` (Duration of video from start time). The 
supported time formats are: 1:13:12, 3:12, 144 (h:min:s, min:s, s). For livestreams just use `--start live` and a duration.

Example: `python rocket_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 19:53 --duration 9:10`

## Installation
Tested with Python 3.9, the package pafy must be installed with: `pip install git+https://github.com/Cupcakus/pafy`

Youtube_dl must be installed with: `pip install youtube_dl`

Tesseract must be installed on the system and referenced, installation link for Windows: https://github.com/UB-Mannheim/tesseract/wiki/Windows-build


## Examples
### Inspiration4
![Velocity plot](examples/inspiration4_velo.png?raw=true)
![Altitude plot](examples/inspiration4_alti.png?raw=true)
![Acceleration plot](examples/inspiration4_acc.png?raw=true)

### Double Asteroid Redirection Test (DART)
![Velocity plot](examples/dart_velo.png?raw=true)
![Altitude plot](examples/dart_alti.png?raw=true)
![Acceleration plot](examples/dart_acc.png?raw=true)

### Transporter-3
![Velocity plot](examples/transporter3_velo.png?raw=true)
![Altitude plot](examples/transporter3_alti.png?raw=true)
![Acceleration plot](examples/transporter3_acc.png?raw=true)