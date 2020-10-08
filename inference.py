import tensorflow as tf
import numpy as np
from model import Encoder, Decoder
import matplotlib.pyplot as plt
import vocabulary
import dataset
import librosa
import hparams


def inference(speaker, text):
    BATCH_SIZE = 1

    encoder = Encoder(BATCH_SIZE, len(vocabulary.VOCAB), len(dataset.SPEAKERS))
    decoder = Decoder(BATCH_SIZE, hparams.n_mel * 4, (hparams.n_fft // 2 + 1) * 4)
    global_step = tf.Variable(0, dtype=tf.int64)  # summary 의 step 은 int64

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder,
                                     global_step=global_step)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("체크포인트 {} 에서 로드합니다...".format(checkpoint_manager.latest_checkpoint))

    print(global_step.numpy(), "step 에서 추론을 진행합니다...")

    tokens = np.array([vocabulary.add_eos(vocabulary.str2token(text))])
    speaker = np.array([dataset.SPEAKER_NAME2IDX[speaker]])

    key, value = encoder(tokens, speaker)
    dec_out, weights, linear = decoder.call_inference(key, value)
    return dec_out[0].numpy().reshape(-1, 80), weights[0].numpy(), linear[0].numpy().reshape(-1, hparams.n_fft // 2 + 1)


if __name__ == '__main__':
    dec_out, weights, linear = inference("kss_0", "다람쥐 헌 쳇바퀴에 타고파, The quick brown fox jumps over the lazy dog.")

    plt.imshow(dec_out)
    plt.show()
    plt.imshow(linear)
    plt.show()
    plt.imshow(weights)
    plt.show()

    stft = linear.T * (hparams.max_db - hparams.min_db) + hparams.min_db  # dB
    stft = np.power(10, stft * 0.05)  # amplitude
    wav = librosa.griffinlim(stft, hop_length=hparams.hop_length)
    wav = librosa.util.normalize(wav)
    librosa.output.write_wav("decoder_output.wav", wav, hparams.sample_rate)
