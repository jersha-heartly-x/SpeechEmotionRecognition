import os

def convert_audio(audio_path, target_path, remove=False):
    # trimming audio from audio_path & saving in target_path with sampling rate 16000 and audio channel 1 (mono) 
    v = os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    if remove:
        os.remove(audio_path)
    return v


