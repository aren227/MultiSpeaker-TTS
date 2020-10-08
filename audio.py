import numpy as np
import librosa
import hparams


def fix_audio(wav, sr):
    aver = np.average(wav[::8])
    wav -= aver
    wav = librosa.util.normalize(wav)

    spl = librosa.effects.split(wav, top_db=30, frame_length=sr // 4,
                                hop_length=sr // 8)
    result = []
    for i in spl:
        result.extend(wav[i[0]:i[1]])
    return np.array(result)


def db_norm(mel):
    return (np.clip(mel, hparams.min_db, hparams.max_db) - hparams.min_db) / (hparams.max_db - hparams.min_db)


def get_linear_and_mel(wav):
    melbank = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, hparams.n_mel)

    stft = np.abs(librosa.stft(wav, hparams.n_fft, hop_length=hparams.hop_length))
    mel = np.dot(melbank, stft ** 2)

    stft_db = 20 * np.log10(np.maximum(stft, 1e-10))
    mel_db = 10 * np.log10(np.maximum(mel, 1e-10))

    stft_db_norm = db_norm(stft_db)
    mel_db_norm = db_norm(mel_db)

    stft_db_norm = stft_db_norm.T  # [time x fft]
    mel_db_norm = mel_db_norm.T  # [time x n_mel]

    if stft_db_norm.shape[0] % 4 != 0:
        stft_db_norm = np.pad(stft_db_norm, [[0, 4 - (stft_db_norm.shape[0] % 4)], [0, 0]], 'constant')
        mel_db_norm = np.pad(mel_db_norm, [[0, 4 - (mel_db_norm.shape[0] % 4)], [0, 0]], 'constant')

    return stft_db_norm, mel_db_norm
