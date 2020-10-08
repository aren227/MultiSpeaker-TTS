import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, LayerNormalization, Activation, Bidirectional, Dropout, GRU, LSTM, Dense, BatchNormalization
import hparams
import math
import vocabulary


class Encoder(tf.keras.Model):

    def __init__(self, batch_size, num_char, num_speaker):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.num_char = num_char
        self.num_speaker = num_speaker

        self.char_embed = Embedding(self.num_char, hparams.char_embed_dim)
        self.speaker_embed = Embedding(self.num_speaker, hparams.speaker_embed_dim)

        self.pre_conv_proj = Dense(hparams.enc_conv_dim)

        self.conv_layers = []
        for i in range(hparams.enc_conv_blocks):
            self.conv_layers.append(ConvBlock())

        self.skip_proj = Dense(hparams.enc_lstm_dim * 2)
        self.lstm = Bidirectional(LSTM(hparams.enc_lstm_dim, return_sequences=True))
        self.lstm_drop = Dropout(hparams.enc_lstm_dropout)
        self.lstm_ln = LayerNormalization()

        self.key_proj = Dense(hparams.query_key_dim)
        self.value_proj = Dense(hparams.value_dim)

    def call(self, char_input, speaker_input):
        x = self.char_embed(char_input)

        speaker = self.speaker_embed(speaker_input)
        speaker = tf.expand_dims(speaker, axis=1)

        x = self.pre_conv_proj(x)

        for i in range(hparams.enc_conv_blocks):
            x = self.conv_layers[i](x, speaker)

        x = self.lstm_drop(self.lstm(x)) + self.skip_proj(x)

        x = self.lstm_ln(x)

        value = self.value_proj(x)
        key = self.key_proj(x)

        return key, value


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ConvBlock, self).__init__()

        self.conv = Conv1D(filters=hparams.enc_conv_dim, kernel_size=5, padding='same', dilation_rate=1)
        self.speaker_proj = Dense(hparams.enc_conv_dim)
        self.ln = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(hparams.enc_conv_dropout)

    def call(self, x, speaker):
        skip = x

        x = self.conv(x)
        x += self.speaker_proj(speaker)

        x = tf.nn.relu(self.ln(x))

        x = self.dropout(x)

        x = x + skip

        return x


class Decoder(tf.keras.Model):

    def __init__(self, batch_size, num_mel, num_linear):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.num_mel = num_mel
        self.num_linear = num_linear

        self.prenet1 = Conv1D(filters=hparams.dec_prenet_dim, kernel_size=5, padding='causal')
        self.prenet1_drop = Dropout(hparams.dec_prenet_dropout)
        self.prenet2 = Dense(hparams.dec_prenet_dim, 'relu')
        self.prenet_ln = LayerNormalization()

        self.query_lstm = LSTM(hparams.dec_prenet_dim, return_sequences=True)
        self.query_lstm_drop = Dropout(hparams.dec_query_lstm_dropout)
        self.query_proj = Dense(hparams.query_key_dim)
        self.skip_proj = Dense(hparams.value_dim)

        self.attention = ScaledDotProductAttention(batch_size)

        self.context_ln = LayerNormalization()

        self.lstm1 = LSTM(hparams.dec_lstm_dim, return_sequences=True)
        self.lstm2 = LSTM(hparams.dec_lstm_dim, return_sequences=True)

        self.mel_ln = LayerNormalization()
        self.mel_proj = Dense(self.num_mel, 'sigmoid')
        self.stop = Dense(1)

        self.post_conv1 = Conv1D(filters=self.num_mel * 2, kernel_size=3, padding='same')
        self.post_conv1_ln = LayerNormalization()
        self.post_conv2 = Conv1D(filters=self.num_linear // 2, kernel_size=3, padding='same')
        self.post_conv3 = Conv1D(filters=self.num_linear, kernel_size=3, padding='same')

    def call(self, mel_input, encoder_key, encoder_value, text_input):
        mel_input = tf.pad(mel_input, [[0, 0], [1, 0], [0, 0]], mode='CONSTANT', constant_values=0)
        mel_input = mel_input[:, :-1, :]

        prenet = self.prenet2(self.prenet1_drop(tf.nn.relu(self.prenet1(mel_input))))
        prenet = self.prenet_ln(prenet)

        query = self.query_lstm_drop(self.query_lstm(prenet)) + prenet

        mask = tf.cast(tf.equal(text_input, vocabulary.CHAR2IDX[vocabulary.PAD]), tf.float32)  # Negative attention to PAD
        mask = tf.expand_dims(mask, 1)  # [B x 1 x Tk]

        context, weight = self.attention(self.query_proj(query), encoder_key, encoder_value, mask)

        context += self.skip_proj(query)  # 중요

        context = self.context_ln(context)

        stop_output = self.stop(context)

        context = self.lstm1(context) + context
        context = self.lstm2(context) + context
        context = self.mel_ln(context)

        mel_output = self.mel_proj(context)

        linear_output = tf.nn.tanh(self.post_conv1_ln(self.post_conv1(mel_output)))
        linear_output = tf.nn.tanh(self.post_conv2(linear_output))
        linear_output = tf.nn.sigmoid(self.post_conv3(linear_output))

        return linear_output, mel_output, stop_output, weight, prenet

    def call_inference(self, encoder_key, encoder_value):
        sos = tf.zeros([self.batch_size, 1, self.num_mel])
        dec_input = sos

        prev_idx = 0.0  # Forcing monotonic

        # Not work for multiple batches
        mask = tf.range(0, tf.shape(encoder_key)[1])
        mask = tf.abs(mask - tf.cast(prev_idx, tf.int32))
        padding_size = 2
        mask = tf.cast(tf.greater(mask, padding_size), tf.float32)
        mask = tf.reshape(mask, [1, 1, -1])

        for i in range(500):  # 반복문 상한
            prenet = self.prenet2(self.prenet1_drop(tf.nn.relu(self.prenet1(dec_input)), training=True))  # 추론시에도 Dropout 적용
            prenet = self.prenet_ln(prenet)

            query = self.query_lstm(prenet) + prenet

            context, weight = self.attention(self.query_proj(query), encoder_key, encoder_value, mask)

            prev_idx = tf.minimum(
                tf.maximum(tf.cast(tf.argmax(weight[0, -1], output_type=tf.int32), tf.float32), prev_idx + 0.2),
                prev_idx + 3)

            mask_ = tf.range(0, tf.shape(encoder_key)[1])
            mask_ = tf.abs(mask_ - tf.cast(prev_idx, tf.int32))
            padding_size = 2
            mask_ = tf.cast(tf.greater(mask_, padding_size), tf.float32)
            mask_ = tf.reshape(mask_, [1, 1, -1])
            mask = tf.concat([mask, mask_], axis=1)

            context += self.skip_proj(query)

            context = self.context_ln(context)

            stop_output = self.stop(context)

            context = self.lstm1(context) + context
            context = self.lstm2(context) + context
            context = self.mel_ln(context)

            mel_output = self.mel_proj(context)

            linear_output = tf.nn.tanh(self.post_conv1_ln(self.post_conv1(mel_output)))
            linear_output = tf.nn.tanh(self.post_conv2(linear_output))
            linear_output = tf.nn.sigmoid(self.post_conv3(linear_output))

            dec_input = tf.concat([dec_input, mel_output[:, -1:, :]], axis=1)

            if tf.nn.sigmoid(stop_output[0, -1, 0]) > 0.5:
                break

        dec_input = dec_input[:, 1:, :]

        return dec_input, weight, linear_output


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, batch_size):
        super(ScaledDotProductAttention, self).__init__()

        self.batch_size = batch_size

        self.score_dropout = Dropout(hparams.dec_score_dropout)

    # Positional Encoding 생성
    def create_pos_enc(self, shape, scale):
        # scale is [B, 1]
        base = tf.fill([shape[1], shape[2] // 2], 10000.0)  # [T x C // 2]
        pos_dim = tf.range(0, shape[2] // 2, dtype=tf.float32) / tf.cast(shape[2] // 2, tf.float32)  # [C // 2]
        base = tf.pow(base, pos_dim)  # [T x C // 2]
        pos_idx = tf.range(0, shape[1], dtype=tf.float32)  # [T]
        pos_idx = tf.expand_dims(pos_idx, axis=-1)  # [T x 1]
        pos_idx = tf.tile(pos_idx, [1, shape[2] // 2])  # [T x C // 2]
        angle = pos_idx / base  # [T x C // 2]
        angle = tf.expand_dims(angle, 0)  # [1 x T x C // 2]
        angle = tf.tile(angle, [shape[0], 1, 1])  # [B x T x C // 2]
        scale = tf.expand_dims(scale, 1)  # [B x 1 x 1]
        angle = angle * scale  # [B x T x C // 2]
        sin = tf.sin(angle)
        cos = tf.cos(angle)
        return tf.concat([sin, cos], axis=-1)  # [B x T x C]

    def call(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)  # [B x Tq x Tk]
        score = score / math.sqrt(hparams.query_key_dim)

        score -= mask * 100

        score = tf.nn.softmax(score, axis=-1)  # [B x Tq x Tk]

        score = self.score_dropout(score)

        context = tf.matmul(score, value)  # [B x Tq x Cv]

        return context, score
