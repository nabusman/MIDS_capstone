import argparse

import h5py
from keras.models import load_model


def main(model_path, video_path, sample_rate, sample_unit, dir_path):
    # Load the model
    model = load_model(model_path)
    # for number of frames:
    
    # Extract frame
    # Predict frame
    # Save statistics
    # 
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Process video and extract \
        the number of people per time unit')
    # model file path
    parser.add_argument('-m', '--model_path', required = True, help = 'Path \
        to the Keras h5 model path')
    # video file path
    parser.add_argument('-v', '--video_path', required = True, help = 'Path \
        to the video file to process')
    # sampling rate
    parser.add_argument('-s', '--sample_rate', required = False, default = 1,
        type = int, help = 'The rate to sample time unit, must be an integer, \
        for instance if you want to sample 1 frame per second, put a 1 here')
    # time unit of sampling rate (s, m, h)
    parser.add_argument('-u', '--sample_unit', required = False, default = 's',
        choices = ['s', 'm', 'h'], help = 'The time unit to sample, default is \
        "s" for seconds, options are: s (seconds), m (minutes), h (hours); \
        for instance if you want to sample 1 frame per second, put a "s" here')
    # output directory path
    parser.add_argument('-d', '--dir_path', required = True, help = 'Path \
        to the directory for output')
    args = parser.parse_args()
    main(**args)