from pydub import AudioSegment
from pydub.utils import make_chunks
# from scipy import signal
import librosa
# import librosa.display
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
#rom sklearn import datasets, cluster
#from mpl_toolkits.mplot3d import Axes3D
import argparse
import pickle


# plt.rcParams['figure.figsize'] = (14, 4)
# plt.rcParams['agg.path.chunksize'] = 100000
#%matplotlib inline

BASE_DIR = os.path.abspath(os.getcwd())

SAMPLE_WIDTH = 2
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
FFT_SIZE = 128
FFT_HOP = int(FFT_SIZE * 0.5)
FFW_AMOUNT = 50000
OVERLAP = 0.5

class SkipException(Exception):
    pass


def get_normalized_path(file_path):
    comps = file_path.split('.')
    filebase = '.'.join(comps[:-1])
    return '{}.norm.wav'.format(filebase)

def ffmpeg_decode(file_src, file_dest):
    assert os.system('ffmpeg -i "{}" -ar 44100 -ac 1 -acodec pcm_s16le "{}"'.format(file_src, file_dest)) == 0

def my_decode(file_path, file_dest):
    try:
        ffmpeg_decode(file_path, file_dest)
    except:
        print('[+] normalizing {}'.format(file_path))
        tmp = AudioSegment.from_file(file_path)
        downsampled = tmp.set_sample_width(SAMPLE_WIDTH)
        if downsampled.channels > 1:
            downsampled = downsampled.split_to_mono()[0]
        if downsampled.frame_rate != SAMPLE_RATE:
            downsampled = downsampled.set_frame_rate(SAMPLE_RATE)
        downsampled.export(norm_file_path, format='wav')

def open_song_from_file(file_path):
    # if has norm cache
    norm_file_path = get_normalized_path(file_path)
    if not os.path.isfile(norm_file_path):
        my_decode(file_path, norm_file_path)
    ret = AudioSegment.from_file(norm_file_path)
    return ret


def get_numpy_song_from_file(file_path):
    return np.array(open_song_from_file(file_path).get_array_of_samples())


def extract_features_from_audiobook(ab):
    features = []
    frame_length = int(SAMPLE_RATE * 1)
    hop_length = int(frame_length / 2.0)
    y = get_numpy_song_from_file(ab)
    frames = librosa.util.frame(y=y, frame_length=frame_length, hop_length=hop_length)
    
    f = 0
    for frame in frames.T:
        f += 1
        print('frame {}/{}'.format(f, frames.shape[1]))
        centroid = librosa.feature.spectral_centroid(y=frame,
                                                     sr=SAMPLE_RATE,
                                                     n_fft=FFT_SIZE,
                                                     hop_length=FFT_HOP).mean()
        spec_band = librosa.feature.spectral_bandwidth(y=frame, sr=SAMPLE_RATE).mean()
        contrast = librosa.feature.spectral_contrast(y=frame, sr=SAMPLE_RATE).mean()
        features.append((centroid, spec_band, contrast))
    open(ab + '.features', 'wb').write(pickle.dumps(features))

def train():
    audiobooks = open(os.path.join(BASE_DIR, "audiobooks.win"), "r").read().splitlines()
    features = []
    i = 0
    for ab in audiobooks:
        i+=1
        print('loading feature {}/{}'.format(i, len(audiobooks)))
        f = pickle.loads(open(ab + '.features', 'rb').read())
        features.append(f)
    features = np.concatenate(features)
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    print('[+] MinMaxScaler fit_transform ({} features)'.format(len(features)))
    features_scaled = min_max_scaler.fit_transform(features)
    print(features_scaled.shape)
    print(features_scaled.min(axis=0))
    print(features_scaled.max(axis=0))
    open(os.path.join(BASE_DIR, 'scaler'), 'wb').write(pickle.dumps(min_max_scaler))

    # plt.scatter(features_scaled[:,0], features_scaled[:,1])
    # plt.xlabel('Zero Crossing Rate (scaled)')
    # plt.ylabel('Spectral Centroid (scaled)')

    model = sklearn.cluster.KMeans(n_clusters=3)
    print('[+] KMeans fit ({} features)'.format(len(features_scaled)))
    model.fit(features_scaled)
    print(model.get_params())
    print('[+] writing model')

    open(os.path.join(BASE_DIR, 'model'), 'wb').write(pickle.dumps(model))

def plot():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (14, 4)
    audiobooks = open(os.path.join(BASE_DIR, "audiobooks.win"), "r").read().splitlines()
    features = []
    i = 0
    for ab in audiobooks:
        i+=1
        print('loading feature {}/{}'.format(i, len(audiobooks)))
        f = pickle.loads(open(ab + '.features', 'rb').read())
        features.append(f)
    features = np.concatenate(features)
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    print('[+] MinMaxScaler fit_transform ({} features)'.format(len(features)))
    features_scaled = min_max_scaler.fit_transform(features)
    print(features_scaled.shape)
    print(features_scaled.min(axis=0))
    print(features_scaled.max(axis=0))

    model = pickle.loads(open(os.path.join(BASE_DIR, 'model'), 'rb').read())

    labels = model.predict(features_scaled)
    # import pdb; pdb.set_trace()
    from mpl_toolkits.mplot3d import Axes3D

    # plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
    # plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], features_scaled[labels==0,2], c='r')
    ax.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], features_scaled[labels==1,2], c='g')
    ax.scatter(features_scaled[labels==2,0], features_scaled[labels==2,1], features_scaled[labels==2,2], c='b')
    # plt.xlabel('???')
    # plt.ylabel('???')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend(('Class 0', 'Class 1'))
    plt.show()

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--train', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--extract')
args = parser.parse_args()

def main(args):
    if args.train:
        train()
        return

    if args.plot:
        plot()
        return
    
    if args.extract:
        try:
            done_file = args.extract + '.done'
            if not os.path.isfile(done_file):
                extract_features_from_audiobook(args.extract)    
                open(done_file, 'w').write('done')
            else:
                print('[+] {} Already done.. Next!'.format(os.path.basename(args.extract)))
        except Exception as e:
            print('[!] Skipping {}'.format(args.extract))
            open('skipped.txt', 'a').write('{}: {}\r\n'.format(args.extract, str(e)))
            raise
        return
    parser.print_help()

main(args)