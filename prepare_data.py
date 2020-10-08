import tensorflow as tf
import numpy as np
import vocabulary
import dataset
import pathlib
from multiprocessing import Pool
import librosa
import hparams
import audio
import random


def serialize_example(text, speaker, linear, mel, stop):
    feature = {
        "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
        "speaker": tf.train.Feature(int64_list=tf.train.Int64List(value=speaker)),
        "linear": tf.train.Feature(float_list=tf.train.FloatList(value=linear)),
        "mel": tf.train.Feature(float_list=tf.train.FloatList(value=mel)),
        "stop": tf.train.Feature(float_list=tf.train.FloatList(value=stop))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def process_batches(batches, batch_size, file_name):
    with tf.io.TFRecordWriter(file_name) as writer:
        batch_count = len(batches) // batch_size

        for i in range(batch_count):
            token_maxlen = 0
            for j in range(batch_size):
                token_maxlen = max(token_maxlen, len(batches[i * batch_size + j][1]))

            texts = []
            speakers = []
            for j in range(batch_size):
                texts.append(vocabulary.add_pad(batches[i * batch_size + j][1], token_maxlen - len(batches[i * batch_size + j][1])))  # 패딩처리
                speakers.append(batches[i * batch_size + j][2])

            spec_maxlen = 0
            linears = []
            mels = []
            stops = []
            for j in range(batch_size):
                wav, sr = librosa.load(batches[i * batch_size + j][0], sr=hparams.sample_rate)
                wav = audio.fix_audio(wav, sr)
                linear, mel = audio.get_linear_and_mel(wav)

                linears.append(linear)
                mels.append(mel)
                stops.append(np.zeros([mel.shape[0] - 1, 1], np.float32))  # mel보다 길이가 1 짧게 만들어, 마지막 프레임을 밑에서 1로 패딩처리 => 마지막 프레임에서 stop

                spec_maxlen = max(spec_maxlen, mel.shape[0])

            for j in range(batch_size):
                linears[j] = np.pad(linears[j], [[0, spec_maxlen - linears[j].shape[0]], [0, 0]], 'constant')
                mels[j] = np.pad(mels[j], [[0, spec_maxlen - mels[j].shape[0]], [0, 0]], 'constant')
                stops[j] = np.pad(stops[j], [[0, spec_maxlen - stops[j].shape[0]], [0, 0]], 'constant', constant_values=1.0)

            writer.write(serialize_example(np.array(texts, dtype=np.int).flatten(),
                                           np.array(speakers, dtype=np.int).flatten(),
                                           np.array(linears).flatten(),
                                           np.array(mels).flatten(),
                                           np.array(stops).flatten()))


if __name__ == '__main__':
    data = dataset.get_all_files()

    print("총", len(data), "개의 파일을 찾았습니다.")
    print("화자:", len(dataset.SPEAKERS), "명")

    total_time = 0
    for d in data:
        total_time += d[3]
    print("길이: {:.2f} 시간".format(total_time / 3600))

    for i in range(len(data)):
        data[i][1] = vocabulary.add_eos(data[i][1])

    BATCH_SIZE = hparams.batch_size

    random.shuffle(data)
    data = sorted(data, key=lambda element: element[3])  # 오디오 길이 정렬

    # for i in range(len(data)):
    #     print(vocabulary.token2str(data[i][1]))

    pathlib.Path("dataset").mkdir(exist_ok=True)

    BATCHES_PER_FILE = 10

    queue_data = []
    idx = 0
    while idx * BATCH_SIZE * BATCHES_PER_FILE < len(data):
        queue_data.append((data[idx * BATCH_SIZE * BATCHES_PER_FILE: (idx + 1) * BATCH_SIZE * BATCHES_PER_FILE], BATCH_SIZE, "dataset/data_" + str(idx) + ".tfrecord"))
        idx += 1

    with Pool(8) as pool:
        hours = pool.starmap(process_batches, queue_data)

    print("데이터가 성공적으로 처리되었습니다.")
