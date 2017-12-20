import os
import argparse
import glob
from pydub import AudioSegment
import uuid
from pprint import pprint
import subprocess
import re
import sys

SAMPLE_RATE = 44100
LENGTH = 2
FFT_SIZE = 128


def my_basename(path):
    if '\\' in path:
        return path.split('\\')[-1]
    return path.split('/')[-1]


def split_audio(input_file, output_dir, length):
    sys.stderr.write('split_audio: {}\r\n'.format(my_basename(input_file)))
    ext = input_file.split('.')[-1]
    name = 'out_' + my_basename(input_file).replace(' ', '_')[:-(len(ext)+1)]
    out_file = '{}%05d.{}'.format(os.path.join(output_dir, name), ext)
    cmd = 'ffmpeg -i "{}" -f segment -segment_time {} -c copy "{}"'.format(input_file,
                                                                           length,
                                                                           out_file)
    # assert os.system(cmd) != 1
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # import pdb; pdb.set_trace())
    # out_files = glob.glob(os.path.join(output_dir, name+'*'))
    if p.returncode != 0:
        print(out.decode('utf-8'))
        print(err.decode('utf-8'))
        raise RuntimeError('Did not go smoothly..', out, err)
    # out_files = re.findall('Opening \'(.*)\' for writing', err.decode('utf8'))
    # for f in out_files:
    #     print(f)
    # print(out.decode('utf-8'))
    # print(err.decode('utf-8'))


def mix_audio(input_nonvoc, input_voc, output_dir):
    sys.stderr.write('mix_audio: {} + {}\r\n'.format(os.basename(input_nonvoc), 
                                                     os.basename(input_voc)))
    nonvoc = AudioSegment.from_file(input_nonvoc)
    voc = AudioSegment.from_file(input_voc)

    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = nonvoc.overlay(voc, position=0)
    output_filename = str(uuid.uuid4())
    # save the result
    # voc.export(os.path.join(output_dir, '{}_gt.wav'.format(output_filename)), format='wav')
    out_file = os.path.join(output_dir, '{}.wav'.format(output_filename))
    output.export(out_file, format='wav')
    print('"{}" "{}"'.format(input_voc, out_file))


def measure_db(input_file):
    i = AudioSegment.from_file(input_file)
    print(i.dBFS)


def filter_db(input_file, threshold):
    i = AudioSegment.from_file(input_file)
    if threshold > i.dBFS:
        print('BELOW_THRESHOLD', input_file)
    else:
        print('ABOVE_THRESHOLD', input_file)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('nonvoc', type=str)
    # parser.add_argument('voc', type=str)

    parser.add_argument('audiofiles', nargs='+')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--measuredb', action='store_true', default=False)
    parser.add_argument('--filterdb', action='store_true', default=False)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--mix', action='store_true', default=False)
    args = parser.parse_args()

    if args.split:
        assert len(args.audiofiles) == 1
        input_file = args.audiofiles[0]
        output_dir = args.out_dir
        split_audio(input_file, output_dir, LENGTH)
    elif args.mix:
        assert len(args.audiofiles) == 2
        input_nonvoc_file = args.audiofiles[0]
        input_voc_file = args.audiofiles[1]
        output_dir = args.out_dir
        mix_audio(input_nonvoc_file, input_voc_file, output_dir)
    elif args.measuredb:
        assert len(args.audiofiles) == 1
        input_file = args.audiofiles[0]
        measure_db(input_file)
    elif args.filterdb:
        assert len(args.audiofiles) == 1
        input_file = args.audiofiles[0]
        filter_db(input_file, args.threshold)

if __name__ == '__main__':
    main()

# split_audio('/Users/amiramitai/Projects/nomix/looperman-a-0047094-0000823-theacidizer-i-dont-know-why-i-even-try.mp3', '/tmp/')
# mix_audio('/tmp/out_looperman-a-0047094-0000823-theacidizer-i-dont-know-why-i-even-try00000.mp3', '/tmp/out_looperman-a-0047094-0000823-theacidizer-i-dont-know-why-i-even-try00001.mp3', '/tmp/')