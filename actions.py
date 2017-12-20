from pydub import AudioSegment

def print_info(songs):
    print('[+] print info!')
    for song in songs:
        print('frame_rate', song.frame_rate)
        print('frames', int(song.frame_count()))
        print('sample_width', int(song.sample_width))
        print('channels', int(song.channels))


def do_action(args):
    if args.action == 'info':
        print_info(args)
        return

    print('[!] no such action', args.action)