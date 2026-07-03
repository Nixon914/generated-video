import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models, initializers
import tensorflow_datasets as tfp
import tensorflow as tf
def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1):
    layers_list = []
    # Depthwise convolution
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
        
        # 計算滑動窗口的相關性，最大偏移量為 max_displacement
        for y_disp in range(-self.max_displacement, self.max_displacement + 1):
            for x_disp in range(-self.max_displacement, self.max_displacement + 1):
                shifted_input_2 = tf.roll(input_2, shift=[y_disp, x_disp], axis=[1, 2])
                corr = tf.reduce_sum(input_1 * shifted_input_2, axis=-1, keepdims=True)
                corr_tensors.append(corr)
        
        # 將所有偏移量的相關性張量合併
        corr_map = tf.concat(corr_tensors, axis=-1)
        return corr_map
class FlowNetSConv(tf.keras.Model):
    def __init__(self, batch_norm=True):
        super(FlowNetSConv, self).__init__()

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
        # 如果輸入是單一圖像，則複製它
        if inputs.shape[-1] == 2:
            image1, image2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        else:
            # 如果是單通道輸入，複製它
            image1 = tf.concat([inputs, inputs], axis=-1)
            image2 = image1
        
        # 提取每個圖像的特徵
        conv1_img1 = self.conv1(image1)
        conv1_img2 = self.conv1(image2)
        
        conv2_img1 = self.conv2(conv1_img1)
        conv2_img2 = self.conv2(conv1_img2)

        conv3_img1 = self.conv3(conv2_img1)
        conv3_img2 = self.conv3(conv2_img2)

        # 計算相關性
        corr = self.corr(conv3_img1, conv3_img2)
        
        # 重定向卷積
        redir = self.conv_redir(conv3_img1)

        # 合併相關性和重定向特徵
        concat = tf.concat([corr, redir], axis=-1)

        # 後續卷積
        out_conv3_1 = self.conv3_1(concat)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv3_1, out_conv4, out_conv5, out_conv6

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

        # 使用 CIFAR-10 的輸入尺寸
        flow2_upsampled = tf.image.resize(flow2, size=(32, 32), method='bilinear')

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

def get_encoder(latent_dim=64):
    # 修改為 CIFAR-10 的輸入尺寸
    encoder_inputs = keras.Input(shape=(32, 32, 3))
    # 複製輸入的通道數也需要修改
    first_half = encoder_inputs[:, :, :, :3]  # 取前三個通道
    second_half = encoder_inputs[:, :, :, :3]  # 再次取前三個通道
    inputs = tf.concat([first_half, second_half], axis=-1)  # 連接得到 6 通道
    
    encode = FlowNetSConv()
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
    
    decode = FlowNetSDeconv()
    decoder_outputs = decode(decoder_inputs[-1], decoder_inputs)

    return keras.Model(decoder_inputs, decoder_outputs, name='decoder')

def get_vqvae(latent_dim=64, num_embedding=256):
    vq_layer = VectorQuantizer(num_embedding, latent_dim, name='vector_quantizer')
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    # 修改為 CIFAR-10 的輸入尺寸
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
    def __init__(self, train_variance, latent_dim = 32, num_embedding = 128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embedding = num_embedding
        
        self.vqvae = get_vqvae(self.latent_dim, self.num_embedding)
        self.total_loss_tracker = keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name = 'reconstruction_loss'
        )
        self.vq_loss_tracker = keras.metrics.Mean(name = 'vq_loss')
    @property # 裝飾器
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x): # 程式中沒被呼叫到，但是在fit中會被使用
        with tf.GradientTape() as tape: # 用來記錄前向傳播中的操作
            #reconstructions=L1
            reconstructions = self.vqvae(x)

            reconstruction_loss = (
                tf.reduce_mean(tf.abs(x - reconstructions)) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables) # 利用前面紀錄的數值進行梯度損失計算
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss) # 針對total_loss進行跌加計算平均值，並且可進行追蹤
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return{
            'total_loss' : self.total_loss_tracker.result(),
            'accuracy': 100 * (1 - reconstruction_loss)  # 簡單的準確率估計
        }

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# 數據預處理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_scaled = (x_train / 255.0) - 0.5  # 平移中心化
x_test_scaled = (x_test / 255.0) - 0.5

data_variance = np.var(x_train / 255.0)

vqvae_trainer = VQVAETrainer(data_variance, latent_dim = 64, num_embedding = 128)
vqvae_trainer.compile(optimizer = keras.optimizers.Adam(),loss='total_loss',metrics = 'acc')
print(x_train.shape)

history = vqvae_trainer.fit(x_train_scaled, epochs=30, batch_size=128)
plt.figure(figsize=(15, 5))

# 損失圖
plt.subplot(1, 2, 1)
plt.plot(history.history['total_loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()