from utils import get_best_estimators
from emotion_recognition import EmotionRecognizer

from scipy.io import wavfile
from audiomentations import Compose, AddGaussianNoise

import pyaudio
import wave
import numpy as np
from array import array
from sys import byteorder
from struct import pack


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)

    return sample_width, r

def record_to_file(path, noise = False):
    sample_width, data = record()
    
    if(noise):
        augmenter = Compose(
            [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]
        )

        for i in range(5):
            data = np.array(data)
            output_file_path = "gnoise.wav"
            augmented_samples = augmenter(samples=data, sample_rate=RATE)
            wavfile.write(output_file_path, rate=RATE, data=augmented_samples)

    else:
        data = pack('<' + ('h'*len(data)), *data)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()

if __name__ == "__main__":
    estimators = get_best_estimators()
    estimators_str, estimator_dict = get_estimators_name(estimators)
    
    features = ["mfcc", "chroma", "mel"]

    detector = EmotionRecognizer(None, emotions=["sad", "neutral", "happy"], features=features, verbose=0)
    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
    
    print("Please talk...")
    
    # filename1 = "test.wav"
    
    # record_to_file(filename1)
    # result = detector.predict(filename1)
    # print("*** PREDICTION ***")
    # print("You are", result)
    # print('*'*18)
    
    filename2 = "gnoise.wav"
    record_to_file(filename2, noise = True)
    result = detector.predict(filename2)
    print("*** PREDICTION ***")
    print("You are", result)
    print('*'*18)

