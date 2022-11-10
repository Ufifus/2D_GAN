import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras import models, layers, Sequential, backend, optimizers

from __init__ import databuilder_configs, modelbuilder_configs


img_size = [int(dim) for dim in databuilder_configs['img_shape'].split(',')]
print(img_size)


class GAN_2D:
    def __init__(self):
        """
        height: height image in input size
        width: wigth image in input size
        channels: RGB constant
        gen_input_dim: generator input shape with random noise
        """
        self.img_height = img_size[0]
        self.img_wigth = img_size[1]
        print(self.img_height, self.img_wigth)
        self.channels = int(modelbuilder_configs['channels'])
        self.gen_input_dim = int(modelbuilder_configs['latent_dim'])
        self.batch_size = int(modelbuilder_configs['batch_size'])

        """
        dataset_path: path to saved numpy dataset
        generator = model of generation images
        discriminator = model of recognizing fake images and real
        """

        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = optimizers.RMSprop(lr=0.00005)

        self.dataset_path = os.path.join(databuilder_configs['save_path'], 'dataset.npz')
        self.saved_model = os.path.join(databuilder_configs['save_path'], 'Generator')
        self.images = os.path.join(databuilder_configs['save_path'], '../images')

        self.generator = self.build_generator()

        z = layers.Input(shape=(self.gen_input_dim,))
        img = self.generator(z)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.discriminator.trainable = False
        valid = self.discriminator(img)

        self.combined = models.Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])



    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)


    def build_generator(self):
        """Build generator:
        Input shape: (gen_init_dim,)
        Output shape: (None, img_height, img_width, channels)
        """
        model = Sequential()
        model.add(layers.Dense(self.img_height * self.img_wigth * 8, use_bias=False, input_shape=(self.gen_input_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((int(self.img_height / 4), int(self.img_wigth / 4), 128)))
        assert model.output_shape == (None, int(self.img_height / 4), int(self.img_wigth / 4), 128)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.img_height / 4), int(self.img_wigth / 4), 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.img_height / 2), int(self.img_wigth / 2), 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.img_height, self.img_wigth, self.channels)

        print(model.summary())

        return model

    def test_generator(self):
        """Input vector (1, self.gen_input_data) shape and return generators  img"""
        #noise = np.random.rand(self.gen_input_dim)
        noise = (tf.random.normal([1, self.gen_input_dim]) + 1) / 2
        print(noise.shape)
        generated_img = self.generator(noise, training=False)
        print(generated_img)
        plt.imshow(generated_img[0])
        plt.show()
        return generated_img


    def build_discriminator(self):
        """Create discriminar for detect fake images whats made of generator
        Input_shape: (None, img_height, img_width, channels)
        Output_shape: (1), True or False
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[self.img_height, self.img_wigth, self.channels]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        print(model.summary())

        return model


    def test_discriminator(self, img):
        """Input image for analise fake or not"""
        decision = self.discriminator(img)
        print(decision)


    def trainer(self, epochs, sample_interval=50):

        # Load the dataset
        with np.load(os.path.join(databuilder_configs['save_path'], 'dataset.npz')) as dataset:
            X_train = dataset['X_train']
            dataset.close()

        # Rescale -1 to 1

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.gen_input_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save(self.saved_model)


    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.images, f"mnist_{epoch}.png"))

        plt.close()


if __name__ == '__main__':
    gan2d = GAN_2D()
    gan2d.trainer(200, 50)