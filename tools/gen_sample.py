import pyaudio
import sys
import numpy as np
import time
import librosa
import os
import pickle
import argparse

from pydub import AudioSegment
from pydub.utils import make_chunks

BASE_DIR = os.path.abspath(os.getcwd())

SAMPLE_WIDTH = 2
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
FFT_SIZE = 128
FFT_HOP = int(FFT_SIZE * 0.5)
FFW_AMOUNT = 50000
OVERLAP = 0.5

frame_length = int(SAMPLE_RATE * 1)
hop_length = int(frame_length / 2.0)


def get_normalized_path(file_path):
    comps = file_path.split('.')
    filebase = '.'.join(comps[:-1])
    return '{}.norm.wav'.format(filebase)


def ffmpeg_decode(file_src, file_dest):
    assert os.system('ffmpeg -i "{}" -ar 44100 -ac 1 -acodec pcm_s16le "{}"'.format(file_src, file_dest)) == 0


def my_decode(file_path, file_dest):
    ffmpeg_decode(file_path, file_dest)


def open_song_from_file(file_path, norm=False):
    # if has norm cache
    print(norm)
    if not norm:
        tmp = AudioSegment.from_file(file_path)
        if tmp.sample_width != SAMPLE_WIDTH:
            print('[+] changing sample width')
            tmp = tmp.set_sample_width(SAMPLE_WIDTH)
        if tmp.channels > 1:
            print('[+] changing num channels')
            tmp = tmp.split_to_mono()[0]
        if tmp.frame_rate != SAMPLE_RATE:
            print('[+] changing sample rate')
            tmp = tmp.set_frame_rate(SAMPLE_RATE)
        return tmp
    norm_file_path = get_normalized_path(file_path)
    if not os.path.isfile(norm_file_path):
        my_decode(file_path, norm_file_path)
    ret = AudioSegment.from_file(norm_file_path)
    return ret


def get_numpy_song_from_file(file_path, norm=False):
    return np.array(open_song_from_file(file_path, norm=norm).get_array_of_samples())


class AudioEngine:
    def __init__(self):
        self.total_frames = 1

        # PLAYER
        self.data_ptr = None
        self.player = None
        self.stream = None
        self.is_playing = False
        self.start_time = time.time()
        self.start_frame = 0
        self.model = pickle.loads(open(os.path.join(BASE_DIR, 'model'), 'rb').read())
        self.scaler = pickle.loads(open(os.path.join(BASE_DIR, 'scaler'), 'rb').read())

    def init_stream(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.player:
            self.player.terminate()
            self.player = None
        self.data_ptr = self.start_frame * CHANNELS
        self.player = pyaudio.PyAudio()
        self.start_time = time.time()

        self.stream = self.player.open(format=self.player.get_format_from_width(SAMPLE_WIDTH),
                                       channels=CHANNELS,
                                       rate=SAMPLE_RATE,
                                       output=True,
                                       stream_callback=self.stream_callback)

    def stream_callback(self, sin_data, frame_count, time_info, status):
        # print('[+] stream_callback')
        if self.track is None:
            return (None, pyaudio.paComplete)

        if not self.is_playing:
            return (None, pyaudio.paComplete)

        dptr = self.data_ptr
        data = self.track[dptr:dptr + (frame_count * CHANNELS)]
        if len(data) < frame_count * CHANNELS:
            # print('[+] end of stream detected!')
            self.jump_to_frame(0)
            self.is_playing = False
            return (data, pyaudio.paComplete)
        self.data_ptr += len(data)
        time_passed = time.time() - self.start_time
        cursor = (self.start_frame + (time_passed * SAMPLE_RATE)) / self.total_frames
        self.time_update_callback(self.start_frame + int(time_passed * SAMPLE_RATE))
        return (data, pyaudio.paContinue)

    def time_update_callback(self, frame):
        sindex = int(int(len(self.s.T) * (frame / (len(self.track)-1))))

        spec_slice = self.s.T[sindex:sindex+100].T
        centroid = librosa.feature.spectral_centroid(S=spec_slice).mean()
        spec_band = librosa.feature.spectral_bandwidth(S=spec_slice).mean()
        contrast = librosa.feature.spectral_contrast(S=spec_slice).mean()
        feat = self.scaler.transform([(centroid, spec_band, contrast)])
        print(self.model.predict(feat))

    def play(self):
        self.last_play_time = time.time()
        self.track = get_numpy_song_from_file(sys.argv[1])
        S, phase = librosa.magphase(librosa.stft(y=self.track,
                                                 n_fft=FFT_SIZE,
                                                 hop_length=FFT_HOP))
        self.s = S
        self.p = phase
        self.is_playing = True
        self.init_stream()

    def should_stop(self):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audiofiles', nargs=2)
    parser.add_argument('--norm', action='store_true', default=False, help='normalizes input audio')
    parser.add_argument('--num', '-n', type=int, default=1)
    parser.add_argument('--outpath', '-o', type=str, default='.')
    args = parser.parse_args()
    
    print(args.norm)
    print(open_song_from_file(args.audiofiles[0], norm=args.norm))
    # ae = AudioEngine()


if __name__ == '__main__':
    main()
