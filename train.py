import tensorflow as tf
from model import Encoder, Decoder
import glob
import vocabulary
import dataset
import random
import hparams
import plot
import pathlib
import numpy as np


def parse_example(bytes):
    example = tf.io.parse_single_example(
        bytes,
        {"text": tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
         "speaker": tf.io.FixedLenFeature([hparams.batch_size], dtype=tf.int64),
         "linear": tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
         "mel": tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
         "stop": tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)}
    )
    example["text"] = tf.reshape(example["text"], [hparams.batch_size, -1])
    example["linear"] = tf.reshape(example["linear"], [hparams.batch_size, -1, (hparams.n_fft // 2 + 1) * hparams.reduction_factor])
    example["mel"] = tf.reshape(example["mel"], [hparams.batch_size, -1, hparams.n_mel * hparams.reduction_factor])

    example["stop"] = tf.reshape(example["stop"], [hparams.batch_size, -1, hparams.reduction_factor])
    example["stop"] = tf.reduce_sum(example["stop"], axis=-1)
    example["stop"] = tf.cast(tf.greater(example["stop"], 0), dtype=tf.float32)
    example["stop"] = tf.expand_dims(example["stop"], axis=-1)

    return example


files = glob.glob("./dataset/*.tfrecord")
random.shuffle(files)
ds = tf.data.TFRecordDataset(files).map(parse_example)

encoder = Encoder(hparams.batch_size, len(vocabulary.VOCAB), len(dataset.SPEAKERS))
decoder = Decoder(hparams.batch_size, hparams.n_mel * hparams.reduction_factor, (hparams.n_fft // 2 + 1) * hparams.reduction_factor)
global_step = tf.Variable(0, dtype=tf.int64)  # summary 의 step 은 int64

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    hparams.init_lr,
    decay_steps=hparams.lr_decay_steps,
    decay_rate=hparams.lr_decay_rate,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

checkpoint_dir = "./checkpoint"
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder,
                                 global_step=global_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print("체크포인트 {} 에서 로드합니다...".format(checkpoint_manager.latest_checkpoint))

summary_writer = tf.summary.create_file_writer("./summary")


def image_norm(image):
    return (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))


kld = tf.keras.losses.KLDivergence()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# TensorSpec을 지정해 주어야 함수 호출시마다 그래프가 재생성되는 참사를 막을 수 있음
@tf.function(
    input_signature=[tf.TensorSpec(shape=[hparams.batch_size, None], dtype=tf.int64),
                     tf.TensorSpec(shape=[hparams.batch_size], dtype=tf.int64),
                     tf.TensorSpec(shape=[hparams.batch_size, None, (hparams.n_fft // 2 + 1) * hparams.reduction_factor], dtype=tf.float32),
                     tf.TensorSpec(shape=[hparams.batch_size, None, hparams.n_mel * hparams.reduction_factor], dtype=tf.float32),
                     tf.TensorSpec(shape=[hparams.batch_size, None, 1], dtype=tf.float32)])
def train_step(text, speaker, linear, mel, stop):
    with tf.GradientTape() as tape:
        enc_key, enc_val = encoder(text, speaker, training=True)
        linear_output, mel_output, stop_output, weight_output, query = decoder(mel, enc_key, enc_val, text, training=True)

        mel_loss = tf.reduce_mean(tf.abs(mel_output - mel))
        linear_loss = tf.reduce_mean(tf.abs(linear_output - linear))
        stop_loss = bce(stop, stop_output)

        # Attention Guide
        position = tf.range(0, tf.shape(weight_output)[1] * tf.shape(weight_output)[2])
        position = tf.reshape(position, [1, tf.shape(weight_output)[1], tf.shape(weight_output)[2]])
        position = tf.tile(position, [hparams.batch_size, 1, 1])
        stop_index = tf.reshape(tf.argmax(tf.cast(tf.equal(text, vocabulary.CHAR2IDX[vocabulary.EOS]), tf.int32), axis=-1, output_type=tf.int32), [hparams.batch_size, 1, 1])  # [batch_size x 1 x 1]
        x_position = (position % tf.shape(weight_output)[2]) / stop_index
        y_position = (position // tf.shape(weight_output)[2]) / tf.shape(weight_output)[1]
        position = -tf.pow((x_position - y_position) * hparams.att_guide_pos_factor, 2)
        position = tf.cast(position, tf.float32)

        pad_mask = tf.cast(tf.equal(text, vocabulary.CHAR2IDX[vocabulary.PAD]), tf.float32)  # Negative attention for PAD
        pad_mask = tf.expand_dims(pad_mask, 1)  # [B x 1 x Tk]
        position -= pad_mask * 100

        position = tf.nn.softmax(position, axis=-1)

        att_loss = kld(position, weight_output) * hparams.att_guide_rate * tf.exp(-tf.cast(global_step, tf.float32) / hparams.att_guide_decay_steps)

        loss = mel_loss + linear_loss * 0.5 + stop_loss * 0.05 + att_loss

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    if global_step == 0:
        encoder.summary()
        decoder.summary()

    if global_step % 50 == 0:
        with summary_writer.as_default():
            tf.summary.scalar("loss/loss", loss, step=global_step)
            tf.summary.scalar("loss/mel_loss", mel_loss, step=global_step)
            tf.summary.scalar("loss/linear_loss", linear_loss, step=global_step)
            tf.summary.scalar("loss/stop_loss", stop_loss, step=global_step)
            tf.summary.scalar("loss/att_loss", att_loss, step=global_step)
            tf.summary.scalar("loss/lr", lr_schedule(global_step), step=global_step)
            tf.summary.image("mel_output", image_norm(tf.reshape(mel_output, [hparams.batch_size, -1, hparams.n_mel, 1])), step=global_step)
            tf.summary.image("linear_output", image_norm(tf.reshape(linear_output, [hparams.batch_size, -1, hparams.n_fft // 2 + 1, 1])), step=global_step)
            tf.summary.image("mel_target", image_norm(tf.reshape(mel, [hparams.batch_size, -1, hparams.n_mel, 1])), step=global_step)
            tf.summary.image("att_weight", image_norm(tf.expand_dims(weight_output, axis=-1)), step=global_step)
            tf.summary.image("att_guide", image_norm(tf.expand_dims(position, axis=-1)), step=global_step)
            tf.summary.image("enc_key", image_norm(tf.expand_dims(enc_key, axis=-1)), step=global_step)
            tf.summary.image("enc_val", image_norm(tf.expand_dims(enc_val, axis=-1)), step=global_step)
            tf.summary.image("dec_query", image_norm(tf.expand_dims(query, axis=-1)), step=global_step)

    return loss


test_texts = ["소녀의 그림자가 뵈지 않는 날이 계속될수록 소년의 가슴 한 구석에는 어딘가 허전함이 자리잡는 것이었다.",
              "오늘날까지 인류가 알아 낸 지식은, 한 개인이 한평생 체험을 거듭할지라도 그 몇만 분의 일도 배우기 어려운 것이다.",
              "아이는 곶감이라는 말을 듣고는 지금까지 울던 울음을 뚝 그쳤습니다.",
              "그 아래에서는 은어 떼가 노란 꽃잎을 바라보며 놀다가 어느새 온몸이 노랗게 물듭니다."]
test_texts = [vocabulary.add_eos(vocabulary.str2token(s)) for s in test_texts]

test_speakers = [dataset.SPEAKER_NAME2IDX["nikl_fv01"],
                 dataset.SPEAKER_NAME2IDX["nikl_fv01"],
                 dataset.SPEAKER_NAME2IDX["nikl_fv01"],
                 dataset.SPEAKER_NAME2IDX["nikl_fv01"]]


def test_step(texts, speakers):
    pathlib.Path("./plot").mkdir(exist_ok=True)

    # 배치 크기 1로 테스트
    for i in range(len(texts)):
        key, value = encoder(np.array([texts[i]]), np.array([speakers[i]]))
        mel_out, weights, linear_out = decoder.call_inference(key, value)

        plot.plot_and_save(texts[i], weights[0].numpy(), tf.reshape(mel_out[0], [-1, 80]).numpy().T, global_step.numpy(), "./plot/{}-{}.png".format(global_step.numpy(), i))


# Main training loop
for batch in ds.repeat().shuffle(50):
    batch_loss = train_step(batch["text"], batch["speaker"], batch["linear"], batch["mel"], batch["stop"])

    print(vocabulary.token2str(batch["text"][0].numpy()))

    global_step.assign_add(1)
    if global_step % 200 == 0:
        save_path = checkpoint_manager.save()
    if global_step % 2000 == 0:
        test_step(test_texts, test_speakers)

    print("Step {} Loss {:.4f}".format(global_step.numpy(), batch_loss.numpy()))
