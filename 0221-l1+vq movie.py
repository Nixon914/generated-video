import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models, initializers
import tensorflow_datasets as tfp
import tensorflow as tf
def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1):
    layers_list = []
    layers_list.append(layers.DepthwiseConv2D(
        kernel_size, 
        strides=stride, 
        padding='same',
        use_bias=False,
        kernel_initializer=initializers.he_normal()
    ))
    
    if batch_norm:
        layers_list.append(layers.BatchNormalization())
    layers_list.append(layers.ReLU())
    
    # Pointwise convolution (1x1 convolution)
    layers_list.append(layers.Conv2D(
        out_channels, 
        kernel_size=1,  # 1x1 卷积
        padding='same',
        kernel_initializer=initializers.he_normal()
    ))
    
    if batch_norm:
        layers_list.append(layers.BatchNormalization())
    layers_list.append(layers.ReLU())
    
    return models.Sequential(layers_list)

def predict_flow(in_channels):
    return layers.Conv2D(3, kernel_size=3, padding='same', kernel_initializer=initializers.he_normal())

def deconv(in_channels, out_channels):
    return layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=initializers.he_normal())

def crop_like(tensor1, tensor2):
    return tf.image.resize(tensor1, tf.shape(tensor2)[1:3], method='bilinear')
class CorrelationLayer(tf.keras.layers.Layer):
    def __init__(self, max_displacement=10):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement

    def call(self, input_1, input_2):
        b, h, w, c = input_1.shape
        corr_tensors = []
        
        for y_disp in range(-self.max_displacement, self.max_displacement + 1):
            for x_disp in range(-self.max_displacement, self.max_displacement + 1):
                shifted_input_2 = tf.roll(input_2, shift=[y_disp, x_disp], axis=[1, 2])
                corr = tf.reduce_sum(input_1 * shifted_input_2, axis=-1, keepdims=True)
                corr_tensors.append(corr)
        
        corr_map = tf.concat(corr_tensors, axis=-1)
        return corr_map
class FlowNetCConv(tf.keras.Model):
    def __init__(self, batch_norm=True):
        super(FlowNetCConv, self).__init__()

        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batch_norm, 256, 32, kernel_size=1, stride=1)
        self.corr = CorrelationLayer(max_displacement=10)
        self.conv3_1 = conv(self.batch_norm, 256, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)

    @tf.function
    def call(self, inputs):
        if inputs.shape[-1] == 2:
            image1, image2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        else:
            image1 = tf.concat([inputs, inputs], axis=-1)
            image2 = image1

        conv1_img1 = self.conv1(image1)
        conv1_img2 = self.conv1(image2)
        
        conv2_img1 = self.conv2(conv1_img1)
        conv2_img2 = self.conv2(conv1_img2)

        conv3_img1 = self.conv3(conv2_img1)
        conv3_img2 = self.conv3(conv2_img2)

        corr = self.corr(conv3_img1, conv3_img2)

        redir = self.conv_redir(conv3_img1)

        concat = tf.concat([corr, redir], axis=-1)

        out_conv3_1 = self.conv3_1(concat)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv3_1, out_conv4, out_conv5, out_conv6

class FlowNetCDeconv(tf.keras.Model):
    def __init__(self):
        super(FlowNetCDeconv, self).__init__()

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
        print('Inputs shape:', inputs.shape)
        print('Conv outputs:', [output.shape for output in conv_outputs])

        out_conv2 = conv_outputs[0]
        out_conv3 = conv_outputs[1]
        out_conv4 = conv_outputs[2]
        out_conv5 = conv_outputs[3]

        inputs = conv_outputs[-1]

        flow6 = self.predict_flow6(inputs)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(inputs), out_conv5)

        concat5 = tf.concat([out_conv5, out_deconv5, flow6_up], axis=-1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = tf.concat([out_conv4, out_deconv4, flow5_up], axis=-1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = tf.concat([out_conv3, out_deconv3, flow4_up], axis=-1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = tf.concat([out_conv2, out_deconv2, flow3_up], axis=-1)
        flow2 = self.predict_flow2(concat2)

        flow2_upsampled = tf.image.resize(flow2, size=(32, 32), method='bilinear')

        return flow2_upsampled

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embedding, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_ebedding = num_embedding

        self.beta = beta

        w_init = tf.random_uniform_initializer()
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

            encoding_indices = self.get_code_indices(flattened)
            encodings = tf.one_hot(encoding_indices, self.num_ebedding)
            quantizeds = tf.matmul(encodings, self.embedding, transpose_b = True)

            quantizeds = tf.reshape(quantizeds, input_shape)

            commitment_loss = tf.reduce_mean((tf.stop_gradient(quantizeds) - i) ** 2)
            codebook_loss = tf.reduce_mean((quantizeds - tf.stop_gradient(i)) ** 2)
            self.add_loss(self.beta * commitment_loss + codebook_loss)

            quantizeds = i + tf.stop_gradient(quantizeds - i)
            quantized.append(quantizeds)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embedding)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis = 1, keepdims = True)
            + tf.reduce_sum(self.embedding ** 2, axis = 0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis = 1)
        return encoding_indices

def get_encoder(latent_dim=64):
    encoder_inputs = keras.Input(shape=(32, 32, 3))
    first_half = encoder_inputs[:, :, :, :3]  #取前三個通道
    second_half = encoder_inputs[:, :, :, :3]  #再次取前三個通道
    inputs = tf.concat([first_half, second_half], axis=-1)  #連接得到 6 通道
    
    encode = FlowNetCConv()
    conv_outputs = encode(inputs)

    print("Encoder outputs length:", len(conv_outputs))
    print("Encoder outputs shapes:", [output.shape for output in conv_outputs])

    encoder_outputs = []
    for output in conv_outputs:
        conv_layer = layers.Conv2D(latent_dim, 1, padding="same")(output)
        encoder_outputs.append(conv_layer)
    
    return keras.Model(encoder_inputs, encoder_outputs, name='encoder')

def get_decoder(latent_dim=64):
    encoder = get_encoder(latent_dim)
    encoder_output_shapes = [encoder.output[i].shape[1:] for i in range(len(encoder.output))]
    
    decoder_inputs = [keras.Input(shape=shape) for shape in encoder_output_shapes]
    
    decode = FlowNetCDeconv()
    decoder_outputs = decode(decoder_inputs[-1], decoder_inputs)

    return keras.Model(decoder_inputs, decoder_outputs, name='decoder')

def get_vqvae(latent_dim=64, num_embedding=256):
    vq_layer = VectorQuantizer(num_embedding, latent_dim, name='vector_quantizer')
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(32, 32, 3))
    encoder_output = encoder(inputs)
    print(encoder_output)
    quantized_latents = vq_layer(encoder_output)
    print('encoder', quantized_latents)
    reconstructions = decoder(quantized_latents)
    print('decoder', reconstructions)
    return keras.Model(inputs, reconstructions, name='vq_vae')

get_vqvae().summary()

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embedding=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance 
        self.latent_dim = latent_dim
        self.num_embedding = num_embedding
        self.vqvae = get_vqvae(self.latent_dim, self.num_embedding)
        
        #定義損失追蹤器
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
        self.mae_loss_tracker = keras.metrics.Mean(name='mae_loss')  
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.similarity_tracker = keras.metrics.Mean(name='acc')

    def calculate_similarity(self, original, generated):
        mae = tf.reduce_mean(tf.abs(original - generated))
        similarity = 1.0 - tf.minimum(mae / self.train_variance, 1.0)
        return similarity * 100

    @property
    def metrics(self):
        return [
            self.vq_loss_tracker,
            self.mae_loss_tracker,
            self.total_loss_tracker,
            self.similarity_tracker
        ]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)
            
            mae_loss = tf.reduce_mean(tf.keras.losses.mae(x, reconstructions))
            
            vq_loss = sum(self.vqvae.losses)
            
            total_loss = mae_loss + vq_loss

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        similarity = self.calculate_similarity(x, reconstructions)

        #更新損失追蹤器
        self.vq_loss_tracker.update_state(vq_loss)
        self.mae_loss_tracker.update_state(mae_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.similarity_tracker.update_state(similarity)

        return {
            'loss': self.total_loss_tracker.result(),
            'acc': self.similarity_tracker.result()
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            x = data[0]  # 只取輸入數據
        else:
            x = data
        reconstructions = self.vqvae(x)
    
        # 計算測試集的損失值
        mae_loss = tf.reduce_mean(tf.keras.losses.mae(x, reconstructions))
        vq_loss = sum(self.vqvae.losses)
        total_loss = mae_loss + vq_loss
        similarity = self.calculate_similarity(x, reconstructions)
    
        # 更新損失追蹤器
        self.vq_loss_tracker.update_state(vq_loss)
        self.mae_loss_tracker.update_state(mae_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.similarity_tracker.update_state(similarity)
    
        return {
            'loss': self.total_loss_tracker.result(),
            'acc': self.similarity_tracker.result()
        }

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5

data_variance = np.var(x_train / 255.0)
data_variance1 = np.var(x_test / 255.0)
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=64, num_embedding=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer1 = VQVAETrainer(data_variance1, latent_dim=64, num_embedding=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

# 修改訓練程式碼
history = vqvae_trainer.fit(
    x_train_scaled,
    validation_data=(x_test_scaled, x_test_scaled),
    epochs=30,
    batch_size=128
)

# 繪製訓練過程圖表
plt.figure(figsize=(15, 5))

# 損失圖
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], marker='o', label='Train Loss')
plt.plot(history.history['val_loss'], marker='o', label='Test Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], marker='o', label='Train Accuracy')
plt.plot(history.history['val_acc'], marker='o', label='Test Accuracy')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()