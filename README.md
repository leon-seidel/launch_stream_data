# Plot rocket data from YouTube streams
Using PyTesseract for extracting the telemetry data from SpaceX, Rocket Lab and Arianespace launch streams.


### Arguments
Arguments: `--url` (Video URL), `--start` (Start time in video in seconds), `--end` (End time in video, supported formats:
1:13:12, 3:12, 144 (h:min:s, min:s, s))

Example: `python rocket_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 19:53 --end 28:24`

### Installation
Tested with Python 3.9, the package pafy must be installed with: `pip install git+https://github.com/Cupcakus/pafy`

Youtube_dl must be installed with: `pip install youtube_dl`

Tesseract must be installed on the system and referenced, installation link for Windows: https://github.com/UB-Mannheim/tesseract/wiki/Windows-build
