import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models, initializers
import tensorflow_datasets as tfp
import tensorflow as tf

def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1):
    layers_list = []
    layers_list.append(layers.Conv2D(out_channels, kernel_size, strides=stride, padding='same',
                                      kernel_initializer=initializers.he_normal()))
    if batch_norm:
        layers_list.append(layers.BatchNormalization())
    layers_list.append(layers.ReLU())
    return models.Sequential(layers_list)

def predict_flow(in_channels):
    return layers.Conv2D(2, kernel_size=3, padding='same', kernel_initializer=initializers.he_normal())

def deconv(in_channels, out_channels):
    return layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=initializers.he_normal())

def crop_like(tensor1, tensor2):
    return tf.image.resize(tensor1, tf.shape(tensor2)[1:3], method='bilinear')


class FlowNetSConv(tf.keras.Model):
    def __init__(self, batch_norm=True):
        super(FlowNetSConv, self).__init__()

        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)

    def call(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv2, out_conv3, out_conv4, out_conv5, out_conv6

class FlowNetSDeconv(tf.keras.Model):
    def __init__(self):
        super(FlowNetSDeconv, self).__init__()

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.upsampled_flow5_to_4 = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.upsampled_flow4_to_3 = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.upsampled_flow3_to_2 = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', use_bias=False)

        self.upsample_flow2 = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', use_bias=False)
    def call(self, inputs, conv_outputs):
        print('conv', conv_outputs)
        print('input', inputs.shape)
        # out_conv5, out_conv4, out_conv3, out_conv2 = tf.split(conv_outputs, num_or_size_splits=4, axis=-1)
        out_conv2 = conv_outputs[0]
        out_conv3 = conv_outputs[1]
        out_conv4 = conv_outputs[2]
        out_conv5 = conv_outputs[3]
        print(out_conv2)
        print(out_conv3)
        print(out_conv4)
        print(out_conv5)
        inputs = conv_outputs[4]
        print(inputs)
        flow6 = self.predict_flow6(inputs)
        print(flow6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        print(flow6_up)
        out_deconv5 = crop_like(self.deconv5(inputs), out_conv5)
        print(out_deconv5)

        concat5 = tf.concat([out_conv5, out_deconv5, flow6_up], axis=-1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)
        print(out_deconv4)

        concat4 = tf.concat([out_conv4, out_deconv4, flow5_up], axis=-1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)
        print(out_deconv3)

        concat3 = tf.concat([out_conv3, out_deconv3, flow4_up], axis=-1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)
        print(out_deconv2)

        concat2 = tf.concat([out_conv2, out_deconv2, flow3_up], axis=-1)
        flow2 = self.predict_flow2(concat2)
        print(flow2)
        flow2_upsampled = tf.image.resize(flow2, size=(32, 32), method='bilinear')
        print(flow2_upsampled)
        # 調整通道數以匹配輸入
        flow2_upsampled = layers.Conv2D(3, 1, padding='same')(flow2_upsampled)
    
        return flow2_upsampled
        x = keras.Input(shape=(32,32,3))
        # flow2_upsampled = self.upsample_flow2(flow2)
        # flow2_upsampled = crop_like(flow2_upsampled, x)
        flow2_upsampled = tf.image.resize(flow2, size=(x.shape[2], x.shape[3]), method='bilinear')
        

        return flow2_upsampled

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embedding, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_ebedding = num_embedding

        self.beta = beta

        w_init = tf.random_uniform_initializer() # 初始生成均勻分布之w
        self.embedding = tf.Variable(
            initial_value = w_init(
            shape = (self.embedding_dim, self.num_ebedding), dtype = 'float32'
            ),
            trainable = True,
            name = 'embedding_vqvae',
        )
    def call(self, x):
        print(x)
        quantized = []
        for i in x:
            input_shape = tf.shape(i)
            print("in")
            flattened = tf.reshape(i, [-1, self.embedding_dim])

            encoding_indices = self.get_code_indices(flattened) # 尋找對應索引
            encodings = tf.one_hot(encoding_indices, self.num_ebedding) # 對應嵌入(不只用在text)
            quantizeds = tf.matmul(encodings, self.embedding, transpose_b = True)

            quantizeds = tf.reshape(quantizeds, input_shape)

            commitment_loss = tf.reduce_mean((tf.stop_gradient(quantizeds) - i) ** 2)
            codebook_loss = tf.reduce_mean((quantizeds - tf.stop_gradient(i)) ** 2)
            self.add_loss(self.beta * commitment_loss + codebook_loss) # 會自動將損失加入，不需要手動追蹤

            quantizeds = i + tf.stop_gradient(quantizeds - i)
            quantized.append(quantizeds)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embedding) # matmul為矩陣相乘(內積)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis = 1, keepdims = True)
            + tf.reduce_sum(self.embedding ** 2, axis = 0)
            - 2 * similarity
        ) # 歐基里德距離平方|a - b|^2 = |a|^2 + |b|^2 - 2 * a·b

        encoding_indices = tf.argmin(distances, axis = 1)
        return encoding_indices

def get_encoder(latent_dim = 64):
    encoder_inputs = keras.Input(shape = (32,32,3))
    encode = FlowNetSConv()
    conv_outputs = encode(encoder_inputs)

    # 使用 list 來保存每一層的輸出結果
    encoder_outputs = []
    for output in conv_outputs:
        # 對每個輸出進行 Conv2D 處理
        conv_layer = layers.Conv2D(latent_dim, 1, padding="same")(output)
        encoder_outputs.append(conv_layer)
    return keras.Model(encoder_inputs, encoder_outputs, name = 'encoder')

def get_decoder(latent_dim=64):
    print('get out', get_encoder(latent_dim).output)
    # decoder_inputs = keras.Input(shape=get_encoder(latent_dim).output[0].shape[1:])
    encoder_output_shapes = [get_encoder(latent_dim).output[i].shape[1:] for i in range(5)]

    # 為每個 encoder 輸出創建一個輸入
    decoder_inputs = [keras.Input(shape=shape) for shape in encoder_output_shapes]
    print(decoder_inputs)
    decode = FlowNetSDeconv()
    decoder_outputs = decode(decoder_inputs[-1], decoder_inputs)
    # x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(decoder_inputs)  # 上採樣
    # x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)  # 上採樣
    # decoder_outputs = layers.Conv2DTranspose(1, 3, padding='same')(x)  # 恢復到 (28, 28, 1)

    return keras.Model(decoder_inputs, decoder_outputs, name='decoder')

def get_vqvae(latent_dim = 64, num_embedding = 256):
    vq_layer = VectorQuantizer(num_embedding, latent_dim, name = 'vector_quantizer')
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape = (32,32,3))
    encoder_output = encoder(inputs)
    print(encoder_output)
    quantized_latents = vq_layer(encoder_output)
    print('encoder', quantized_latents)
    reconstructions = decoder(quantized_latents)
    print('decoder', reconstructions)
    return keras.Model(inputs, reconstructions, name = 'vq_vae')

get_vqvae().summary()

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embedding=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance 
        self.latent_dim = latent_dim
        self.num_embedding = num_embedding
        self.vqvae = get_vqvae(self.latent_dim, self.num_embedding)
        
        # 只保留 vq_loss 和 accuracy 追蹤器
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
        self.similarity_tracker = keras.metrics.Mean(name='acc')

    def calculate_similarity(self, original, generated):
        # 計算相似度：1 - 標準化後的 MSE
        mae = tf.reduce_mean(tf.abs(original - generated))
        similarity = 1.0 - tf.minimum(mae / self.train_variance, 1.0)
        return similarity * 100

    @property
    def metrics(self):
        return [
            self.vq_loss_tracker,
            self.similarity_tracker
        ]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            # 前向傳播
            reconstructions = self.vqvae(x)
            
            # 獲取 VQ 損失
            vq_loss = sum(self.vqvae.losses)
            
            # 總損失 (這裡只用 vq_loss)
            total_loss = vq_loss

        # 計算梯度
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # 計算相似度
        similarity = self.calculate_similarity(x, reconstructions)

        # 只更新 vq_loss 和 similarity
        self.vq_loss_tracker.update_state(vq_loss)
        self.similarity_tracker.update_state(similarity)

        return {
            'vq_loss': self.vq_loss_tracker.result(),
            'acc': self.similarity_tracker.result()
        }

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# 數據預處理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_scaled = (x_train / 255.0) - 0.5  # 平移中心化
x_test_scaled = (x_test / 255.0) - 0.5

data_variance = np.var(x_train / 255.0)

vqvae_trainer = VQVAETrainer(data_variance, latent_dim=64, num_embedding=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

history = vqvae_trainer.fit(x_train_scaled, epochs=30, batch_size=128)

# 繪製損失圖
plt.figure(figsize=(10, 4))

# VQ Loss 圖
plt.subplot(1, 2, 1)
plt.plot(history.history['vq_loss'], label='VQ Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VQ Loss vs. Epoch')
plt.legend()

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()