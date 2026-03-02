See Google Sheet for ground truth:

https://docs.google.com/spreadsheets/d/1eLLPuMDX48STOiaAON09nKBa8V3pcoAYuTP_rDYmNAM/edit?gid=0#gid=0

Images are raw, 10-bit from raspiraw, all 640 x 480 but may need to be rotated based on sensor model.

raw.py (contact Eon for questions) requires OpenCV (pip install opencv-python) and Numpy (pip install numpy), images can be simply loaded with load_raw.

The more closely mimick what current happens in Golf Video Processor (GVP) on the Raspberry Pi, run gvp_probe_raw() on frame00009.raw in the sequence to generate a lookup table and then apply that looking table with gvp_load_raw() to load any image in the sequence.