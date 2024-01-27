import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import layers, losses 
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras.models import Model
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print (x_train.shape)
print (x_test.shape)
class Autoencoder(Model):
    def _init_ (self, latent_dim, shape):
        super(Autoencoder, self)._init_()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
         layers.Flatten(),
         layers.Dense(latent_dim, activation= 'relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape),activation='sigmoid'),
            layers.Reshape(shape)
        ])
    def call(self,x):
      encoded=self.encoder(x)
      decoded=self.decoder(encoded)
      return decoded
shape=x_test.shape[1:]
latent_dim=64
autoencoder=Autoencoder(latent_dim,shape)
autoencoder.compile(optimizer='adam',loss=losses.MeanSquaredError())
autoencoder.fit(x_train,x_train,epochs=10,shuffle=True,validation_data=(x_test,x_test))
encoded_imgs=autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n=10
plt.figure(figsize=(20,4))
for i in range(n):
  ax=plt.subplot(2,n,i+1)
  plt.imshow(x_test[i])
  plt.title("orginal")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax=plt.subplot(2,n,i+1+n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
