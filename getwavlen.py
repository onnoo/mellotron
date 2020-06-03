import wave
import contextlib
import os
import argparse

def get_length(filename):
    files = open(filename,'r', encoding='utf-8')
    total_duration = 0.0
    lines = files.readlines()
    for line in lines:
        fname = line.split('|')[0]
        with contextlib.closing(wave.open(fname,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            total_duration += duration
    print(total_duration)
    print(total_duration / 60.0)

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    get_length(args.file)