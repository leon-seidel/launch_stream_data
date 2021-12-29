# Plot rocket data from YouTube videos
Using PyTesseract for extracting the telemetry data from SpaceX, Rocket Lab and Arianespace lauch videos.

Arguments: `--url` (Video URL), `--start` (Start time in video in seconds), `--end` (End time in video, supported formats:
1:13:12, 3:12, 144 (h:min:s, min:s, s))

Example: `python rocket_data_live.py --url https://www.youtube.com/watch?v=JBGjE9_aosc --start 19:53 --end 28:24`