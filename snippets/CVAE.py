def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


@tf.keras.utils.register_keras_serializable('cvae')
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(
        self,
        input_shape: list | tuple,
        latent_dim: int,
        encoder_filters: list | tuple,
        filter_size: int | list | tuple = 3,
        strides: int | list | tuple = 1,
        batchnorm: bool = False,
    ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder, self.decoder = conv2d_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            encoder_filters=encoder_filters,
            filter_size=filter_size,
            strides=strides,
            batchnorm=batchnorm,
            double_latent=True,
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

        
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss, 'elbo': -loss}
    
    def test_step(self, x):
        loss = self.compute_loss(x)
        return {'loss': loss, 'elbo': -loss}
    
    
@tf.keras.utils.register_keras_serializable('cae')
class CAE(tf.keras.Model):
    """Convolutional autoencoder."""

    def __init__(
        self,
        input_shape: list | tuple,
        latent_dim: int,
        encoder_filters: list | tuple,
        filter_size: int | list | tuple = 3,
        strides: int | list | tuple = 1,
        tied_weights=False,
        batchnorm: bool = False,
    ):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder, self.decoder = conv2d_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            encoder_filters=encoder_filters,
            filter_size=filter_size,
            strides=strides,
            batchnorm=batchnorm,
            tied_weights=tied_weights,
        )
        self.use_crossentropy = False
    
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            xp = self.decoder(self.encoder(x))
            cross_entropy = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=xp, labels=x), axis=(1, 2, 3))
            )
            kl_divergence = tf.keras.losses.KLDivergence(x, xp)
            euclidean_norm = tf.reduce_mean(
                tf.math.reduce_euclidean_norm(x - xp, axis=(1, 2))
            )
            loss = cross_entropy if self.use_crossentropy else euclidean_norm
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'cross_entropy': cross_entropy, 'euclidean_norm': euclidean_norm, 'kl_divergence': kl_divergence}
    
    def test_step(self, x):
        xp = self.decoder(self.encoder(x))
        cross_entropy = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=xp, labels=x), axis=(1, 2, 3))
        )
        kl_divergence = tf.keras.losses.KLDivergence(x, xp)
        euclidean_norm = tf.reduce_mean(
            tf.math.reduce_euclidean_norm(x - xp, axis=(1, 2))
        )
        return {'cross_entropy': cross_entropy, 'euclidean_norm': euclidean_norm}