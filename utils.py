from pydub import AudioSegment


def open_song_from_file(file_path):
    song = AudioSegment.from_file(file_path)
    return song