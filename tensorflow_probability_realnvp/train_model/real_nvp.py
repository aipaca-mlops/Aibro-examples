import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

tfd = tfp.distributions
tfb = tfp.bijectors


class AffineCouplingLayer(tfb.Bijector):
    """
    Class to implement the affine coupling layer.
    Complete the __init__ and _get_mask methods according to the instructions above.
    """

    def __init__(self, shift_and_log_scale_fn, mask_type, orientation, **kwargs):
        """
        The class initialiser takes the shift_and_log_scale_fn callable, mask_type,
        orientation and possibly extra keywords arguments. It should call the 
        base class initialiser, passing any extra keyword arguments along. 
        It should also set the required arguments as class attributes.
        """
        super(AffineCouplingLayer, self).__init__(forward_min_event_ndims=3, **kwargs)
        self.shift_and_log_scale_fn = shift_and_log_scale_fn
        self.mask_type = mask_type
        self.orientation = orientation
        

class Squeeze(tfb.Bijector):
    
    def __init__(self, name='Squeeze', **kwargs):
        super(Squeeze, self).__init__(forward_min_event_ndims=3, is_constant_jacobian=True, 
                                      name=name, **kwargs)

    def _forward(self, x):
        input_shape = x.shape
        height, width, channels = input_shape[-3:]
        y = tfb.Reshape((height // 2, 2, width // 2, 2, channels), event_shape_in=(height, width, channels))(x)
        y = tfb.Transpose(perm=[0, 2, 1, 3, 4])(y)
        y = tfb.Reshape((height // 2, width // 2, 4 * channels),
                        event_shape_in=(height // 2, width // 2, 2, 2, channels))(y)
        return y

    def _inverse(self, y):
        input_shape = y.shape
        height, width, channels = input_shape[-3:]
        x = tfb.Reshape((height, width, 2, 2, channels // 4), event_shape_in=(height, width, channels))(y)
        x = tfb.Transpose(perm=[0, 2, 1, 3, 4])(x)
        x = tfb.Reshape((2 * height, 2 * width, channels // 4),
                        event_shape_in=(height, 2, width, 2, channels // 4))(x)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., y.dtype)

    def _forward_event_shape_tensor(self, input_shape):
        height, width, channels = input_shape[-3], input_shape[-2], input_shape[-1]
        return height // 2, width // 2, 4 * channels

    def _inverse_event_shape_tensor(self, output_shape):
        height, width, channels = output_shape[-3], output_shape[-2], output_shape[-1]
        return height * 2, width * 2, channels // 4


class RealNVPMultiScale(tfb.Bijector):
    
    def __init__(self, **kwargs):
        super(RealNVPMultiScale, self).__init__(forward_min_event_ndims=3, **kwargs)

        # First level
        shape1 = (32, 32, 3)  # Input shape
        shape2 = (16, 16, 12)  # Shape after the squeeze operation
        shape3 = (16, 16, 6)  # Shape after factoring out the latent variable
        self.conv_resnet1 = get_conv_resnet(shape1, 64)
        self.conv_resnet2 = get_conv_resnet(shape1, 64)
        self.conv_resnet3 = get_conv_resnet(shape1, 64)
        self.conv_resnet4 = get_conv_resnet(shape2, 128)
        self.conv_resnet5 = get_conv_resnet(shape2, 128)
        self.conv_resnet6 = get_conv_resnet(shape2, 128)
        self.squeeze = Squeeze()
        self.block1 = realnvp_block([self.conv_resnet1, self.conv_resnet2,
                                    self.conv_resnet3, self.conv_resnet4,
                                    self.conv_resnet5, self.conv_resnet6], self.squeeze)

        # Second level
        self.conv_resnet7 = get_conv_resnet(shape3, 128)
        self.conv_resnet8 = get_conv_resnet(shape3, 128)
        self.conv_resnet9 = get_conv_resnet(shape3, 128)
        self.conv_resnet10 = get_conv_resnet(shape3, 128)
        self.coupling_layer1 = AffineCouplingLayer(self.conv_resnet7, 'checkerboard', 0)
        self.coupling_layer2 = AffineCouplingLayer(self.conv_resnet8, 'checkerboard', 1)
        self.coupling_layer3 = AffineCouplingLayer(self.conv_resnet9, 'checkerboard', 0)
        self.coupling_layer4 = AffineCouplingLayer(self.conv_resnet10, 'checkerboard', 1)
        self.block2 = tfb.Chain([self.coupling_layer4, self.coupling_layer3,
                                 self.coupling_layer2, self.coupling_layer1])

    def _forward(self, x):
        h1 = self.block1.forward(x)
        z1, h2 = tf.split(h1, 2, axis=-1)
        z2 = self.block2.forward(h2)
        return tf.concat([z1, z2], axis=-1)
        
    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h2 = self.block2.inverse(z2)
        h1 = tf.concat([z1, h2], axis=-1)
        return self.block1.inverse(h1)

    def _forward_log_det_jacobian(self, x):
        log_det1 = self.block1.forward_log_det_jacobian(x, event_ndims=3)
        h1 = self.block1.forward(x)
        _, h2 = tf.split(h1, 2, axis=-1)
        log_det2 = self.block2.forward_log_det_jacobian(h2, event_ndims=3)
        return log_det1 + log_det2

    def _inverse_log_det_jacobian(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h2 = self.block2.inverse(z2)
        log_det2 = self.block2.inverse_log_det_jacobian(z2, event_ndims=3)
        h1 = tf.concat([z1, h2], axis=-1)
        log_det1 = self.block1.inverse_log_det_jacobian(h1, event_ndims=3)
        return log_det1 + log_det2

    def _forward_event_shape_tensor(self, input_shape):
        height, width, channels = input_shape[-3], input_shape[-2], input_shape[-1]
        return height // 4, width // 4, 16 * channels

    def _inverse_event_shape_tensor(self, output_shape):
        height, width, channels = output_shape[-3], output_shape[-2], output_shape[-1]
        return 4 * height, 4 * width, channels // 16


def get_preprocess_bijector(alpha):
    """
    This function should create a chained bijector that computes the 
    transformation T in equation (7) above.
    This can be computed using in-built bijectors from the bijectors module.
    Your function should then return the chained bijector.
    """
    chain_list = [tfb.Scale(1 - 2*alpha), tfb.Shift(alpha), tfb.Invert(tfb.Sigmoid())]
    return tfb.Chain(chain_list[::-1])


class RealNVPModel(Model):

    def __init__(self, **kwargs):
        super(RealNVPModel, self).__init__(**kwargs)
        self.preprocess = get_preprocess_bijector(0.05)
        self.realnvp_multiscale = RealNVPMultiScale()
        self.bijector = tfb.Chain([self.realnvp_multiscale, self.preprocess])
        
    def build(self, input_shape):
        output_shape = self.bijector(tf.expand_dims(tf.zeros(input_shape[1:]), axis=0)).shape
        self.base = tfd.Independent(tfd.Normal(loc=tf.zeros(output_shape[1:]), scale=1.),
                                    reinterpreted_batch_ndims=3)
        self._bijector_variables = (
            list(self.bijector.variables))
        self.flow = tfd.TransformedDistribution(
            distribution=self.base,
            bijector=tfb.Invert(self.bijector),
        )
        super(RealNVPModel, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        return self.flow

    def sample(self, batch_size):
        sample = self.base.sample(batch_size)
        return self.bijector.inverse(sample)
        
    def _get_mask(self, shape):
        """
        This internal method should use the binary mask functions above to compute
        and return the binary mask, according to the arguments passed in to the
        initialiser.
        """
        if self.mask_type == "channel":
            return channel_binary_mask(num_channels=shape[-1], orientation=self.orientation)
        return checkerboard_binary_mask(shape=shape[1:3], orientation=self.orientation) 
        
    def _forward(self, x):
        b = self._get_mask(x.shape)
        return forward(x, b, self.shift_and_log_scale_fn)

    def _inverse(self, y):
        b = self._get_mask(y.shape)
        return inverse(y, b, self.shift_and_log_scale_fn)

    def _forward_log_det_jacobian(self, x):
        b = self._get_mask(x.shape)
        return forward_log_det_jacobian(x, b, self.shift_and_log_scale_fn)

    def _inverse_log_det_jacobian(self, y):
        b = self._get_mask(y.shape)
        return inverse_log_det_jacobian(y, b, self.shift_and_log_scale_fn)


def realnvp_block(shift_and_log_scale_fns, squeeze):
    """
    This function takes a list or tuple of six conv_resnet models, and an 
    instance of the Squeeze bijector.
    The function should construct the chain of bijectors described above,
    using the conv_resnet models in the coupling layers.
    The function should then return the chained bijector.
    """
    chain_list = []
    mask_types = ["checkrboard", "channel"]
    for i in range(2):
        for j in range(3):
            chain_list.append(AffineCouplingLayer(shift_and_log_scale_fns[i*3: (i+1)*3][j], 
                                                  mask_type=mask_types[i], 
                                                  orientation=j%2))
        chain_list.append(tfb.BatchNormalization())
        chain_list.append(squeeze)
    
    return tfb.Chain(chain_list[:-1][::-1])


def get_conv_resnet(input_shape, filters):
    """
    This function should build a CNN ResNet model according to the above specification,
    using the functional API. The function takes input_shape as an argument, which should be
    used to specify the shape in the Input layer, as well as a filters argument, which
    should be used to specify the number of filters in (some of) the convolutional layers.
    Your function should return the model.
    """
    
    KENEL_SIZE = (3, 3)
    l2_reg = tf.keras.regularizers.l2(5e-5)
    h0 = Input(shape=input_shape, name='h0')
    g1 = Conv2D(kernel_size=KENEL_SIZE, 
                filters=filters, 
                activation='relu', 
                kernel_regularizer=l2_reg, 
                padding='SAME')(h0)
    g2 = BatchNormalization()(g1)
    g3 = Conv2D(kernel_size=KENEL_SIZE, 
                filters=input_shape[-1], 
                activation='relu', 
                name='h1', 
                kernel_regularizer=l2_reg, 
                padding='SAME')(g2)
    g4 = BatchNormalization()(g3)
    h1 = g4 + h0
    
    g1 = Conv2D(kernel_size=KENEL_SIZE, 
                filters=filters, 
                activation='relu', 
                kernel_regularizer=l2_reg, 
                padding='SAME')(h1)
    g2 = BatchNormalization()(g1)
    g3 = Conv2D(kernel_size=KENEL_SIZE, 
                filters=h1.shape[-1], 
                activation='relu', 
                kernel_regularizer=l2_reg, 
                padding='SAME')(g2)
    g4 = BatchNormalization()(g3)
    g5 = g4 + h1
    
    h2 = Conv2D(kernel_size=KENEL_SIZE, 
                filters=2*input_shape[-1], 
                kernel_regularizer=l2_reg, 
                padding='SAME', 
                name='h2')(g5)
    [shift, log_scale] = tf.split(h2, num_or_size_splits=2, axis=-1)
    log_scale = tf.nn.tanh(log_scale)
    
    model = Model(h0, [shift, log_scale])
    return model


def checkerboard_binary_mask(shape, orientation=0):
    height, width = shape[0], shape[1]
    height_range = tf.range(height)
    width_range = tf.range(width)
    height_odd_inx = tf.cast(tf.math.mod(height_range, 2), dtype=tf.bool)
    width_odd_inx = tf.cast(tf.math.mod(width_range, 2), dtype=tf.bool)
    odd_rows = tf.tile(tf.expand_dims(height_odd_inx, -1), [1, width])
    odd_cols = tf.tile(tf.expand_dims(width_odd_inx, 0), [height, 1])
    checkerboard_mask = tf.math.logical_xor(odd_rows, odd_cols)
    if orientation == 1:
        checkerboard_mask = tf.math.logical_not(checkerboard_mask)
    return tf.cast(tf.expand_dims(checkerboard_mask, -1), tf.float32)


def channel_binary_mask(num_channels, orientation=0):
    """
    This function takes an integer num_channels and orientation (0 or 1) as
    arguments. It should create a channel-wise binary mask with 
    dtype=tf.float32, according to the above specification.
    The function should then return the binary mask.
    """
    
    former_half_channels = num_channels // 2
    later_half_channels = num_channels - former_half_channels
    former_mask = tf.cast(tf.zeros(former_half_channels), tf.bool)
    later_mask = tf.cast(tf.ones(later_half_channels), tf.bool)
    channel_mask = tf.concat([former_mask, later_mask], axis=0)
    if orientation == 1:
        channel_mask = tf.reverse(channel_mask, axis=[0])
    channel_mask = tf.cast(channel_mask, tf.float32)
    
    return tf.reshape(channel_mask, [1, 1, num_channels])


def forward(x, b, shift_and_log_scale_fn):
    """
    This function takes the input Tensor x, binary mask b and callable
    shift_and_log_scale_fn as arguments.
    This function should implement the forward transformation in equation (5)
    and return the output Tensor y, which will have the same shape as x
    """
    y_d = x * b
    shift, log_scale = shift_and_log_scale_fn(y_d)
    y_D = (1-b) * (x * tf.exp(log_scale) + shift)
    y = y_d + y_D
    
    return y


def inverse(y, b, shift_and_log_scale_fn):
    """
    This function takes the input Tensor x, binary mask b and callable
    shift_and_log_scale_fn as arguments.
    This function should implement the forward transformation in equation (5)
    and return the output Tensor y, which will have the same shape as x
    """
    x_d = y * b
    shift, log_scale = shift_and_log_scale_fn(x_d)
    x_D = (1-b) * ((y - shift) * tf.exp(-log_scale))
    x = x_d + x_D
    
    return x


def forward_log_det_jacobian(x, b, shift_and_log_scale_fn):
    """
    This function takes the input Tensor x, binary mask b and callable
    shift_and_log_scale_fn as arguments.
    This function should compute and return the log of the Jacobian determinant 
    of the forward transformation in equation (5)
    """
    _, log_scale = shift_and_log_scale_fn(x * b)
    log_det_jacobian = tf.reduce_sum((1-b)*log_scale, [1, 2, 3])
    
    return log_det_jacobian
    

def inverse_log_det_jacobian(y, b, shift_and_log_scale_fn):
    """
    This function takes the input Tensor y, binary mask b and callable
    shift_and_log_scale_fn as arguments.
    This function should compute and return the log of the Jacobian determinant 
    of the forward transformation in equation (6)
    """
    _, log_scale = shift_and_log_scale_fn(y * b)
    log_det_jacobian = -tf.reduce_sum((1-b)*log_scale, [1, 2, 3])

    return log_det_jacobian


