import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense,\
    GlobalAveragePooling2D, Layer
from tensorflow.keras.layers import Activation, BatchNormalization, Add,\
    Multiply, Reshape, AveragePooling2D
from tensorflow.image import ResizeMethod


def relu6(x):
    """Relu 6."""
    return tf.nn.relu(x)


def hard_swish(x):
    """Hard swish."""
    return x * tf.nn.relu(x + 3.0) / 6.0


def return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.
    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)
    return x


class ConvBlock(Layer):
    """Convolution Block
    This class defines a 2D convolution operation with BN and activation.
    # Arguments
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window. Default=(3,3)
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and
                height. Can be a single integer to specify the same value for
                all spatial dimensions. Default=1
            nl: String, nonlinearity activation type. Default='RE'
            channel_axis: 1 if channels are first in the image and -1 if the
                last. Default=-1
            padding_scheme: Padding scheme to apply for convolution.
                Default='same'
        # call
            x: Tensor, input tensor of conv layer.
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    """

    def __init__(self, filters, kernel=(3, 3), strides=1, nl='RE',
                 padding='same', channel_axis=-1):
        super(ConvBlock, self).__init__()
        self.channel_axis = channel_axis
        self.nl = nl
        self.conv = Conv2D(filters, kernel, padding=padding, strides=strides)
        self.bn = BatchNormalization(axis=channel_axis)

    def call(self, x, training=True):
        x = self.conv(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn(x, training=training)
        return return_activation(x, self.nl)


class Squeeze(Layer):
    """Squeeze and Excitation.
    This function defines a squeeze structure.
    # Arguments
        #call
            inputs: Tensor, input tensor of conv layer
    # Returns
        Output tensor.
    """

    def __init__(self):
        super(Squeeze, self).__init__()

    def build(self, input_shape):
        # print(input_shape)
        self.input_channels = input_shape[-1]
        self.fc1 = Dense(self.input_channels, activation='relu')
        self.fc2 = Dense(self.input_channels, activation='hard_sigmoid')

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = Reshape((1, 1, self.input_channels))(x)
        x = Multiply()([inputs, x])
        return x


class Bottleneck(Layer):
    """Bottleneck
    This class defines a basic bottleneck structure.
    # Arguments
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            expansion: Integer, expansion factor.
                t is always applied to the input size.
            strides: An integer or tuple/list of 2 integers,specifying the
                strides of the convolution along the width and height.
                Can be a single integer to specify the same value for all
                spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.
            alpha: Multiplier of number of intermediate channels
            channel_axis: 1 if channels are first in the image and -1 if
                the last. Default=-1
        #call
            inputs: Tensor, input tensor of conv layer
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    """

    def __init__(self, filters, kernel, expansion, strides, squeeze, nl,
                 alpha=1.0, channel_axis=-1):
        super(Bottleneck, self).__init__()
        self.strides = strides
        self.filters = filters
        self.nl = nl
        self.squeeze = squeeze
        if self.squeeze:
            self.squeeze_layer = Squeeze()
        tchannel = int(expansion)
        cchannel = int(alpha * filters)

        self.conv_block = ConvBlock(tchannel, kernel=(1, 1),
                                    strides=(1, 1), nl=nl)
        self.dw_conv = DepthwiseConv2D(kernel, strides=(strides, strides),
                                       depth_multiplier=1, padding='same')
        self.bn1 = BatchNormalization(axis=channel_axis)
        self.conv2d = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization(axis=channel_axis)

    def build(self, input_shape):
        self.r = self.strides == 1 and input_shape[3] == self.filters

    def call(self, inputs, training=True):
        x = self.conv_block(inputs)
        x = self.dw_conv(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn1(x, training=training)
        x = return_activation(x, self.nl)
        if self.squeeze:
            x = self.squeeze_layer(x)
        x = self.conv2d(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn2(x, training=training)
        if self.r:
            x = Add()([x, inputs])
        return x


class MobileNetV3SmallBackbone(Layer):
    def __init__(self, alpha=1.0, mode='segmentation'):
        """MobileNetV3SmallBackbone.
        # Arguments
            # init
                alpha: Integer, width multiplier.
                mode: String, either "classification" or "segmentation"
            #call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            segm_features1: Feature map with h/8 resolution
            segm_features2: Feature map with h/16 resolution
        """
        super(MobileNetV3SmallBackbone, self).__init__()
        self.alpha = alpha
        self.mode = mode
        self.first_conv = ConvBlock(16, (3, 3), strides=2, nl='HS')  # h/2
        self.bottleneck1 = Bottleneck(
            16, (3, 3), expansion=16, strides=2, squeeze=True,
            nl='RE', alpha=alpha)  # h/4
        self.bottleneck2 = Bottleneck(
            24, (3, 3), expansion=72, strides=2, squeeze=False,
            nl='RE', alpha=alpha)  # h/8
        self.bottleneck3 = Bottleneck(
            24, (3, 3), expansion=88, strides=1, squeeze=False,
            nl='RE', alpha=alpha)  # h/8
        self.bottleneck4 = Bottleneck(
            40, (5, 5), expansion=96, strides=2, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck5 = Bottleneck(
            40, (5, 5), expansion=240, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck6 = Bottleneck(
            40, (5, 5), expansion=240, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck7 = Bottleneck(
            48, (5, 5), expansion=120, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck8 = Bottleneck(
            48, (5, 5), expansion=144, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        if self.mode == 'classification':
            self.bottleneck9 = Bottleneck(
                96, (5, 5), expansion=288, strides=2, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            self.bottleneck10 = Bottleneck(
                96, (5, 5), expansion=576, strides=1, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            self.bottleneck11 = Bottleneck(
                96, (5, 5), expansion=576, strides=1, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            # Last stage
            self.last_stage_conv1 = ConvBlock(
                576, (1, 1), strides=1, nl='HS')  # h/32
            self.last_stage_conv2 = Conv2D(1280, (1, 1), padding='same')  # h/h

    def call(self, inputs, training=True):
        # print(inputs.shape)
        x = self.first_conv(inputs, training=training)
        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        segm_features1 = self.bottleneck3(x, training=training)
        x = self.bottleneck4(segm_features1, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)
        x = self.bottleneck7(x, training=training)
        segm_features2 = self.bottleneck8(x, training=training)
        if self.mode == 'segmentation':
            return segm_features1, segm_features2

        elif self.mode == 'classification':
            x = self.bottleneck9(segm_features2, training=training)
            x = self.bottleneck10(x, training=training)
            x = self.bottleneck11(x, training=training)
            # Last stage
            x = self.last_stage_conv1(x, training=training)
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, 576))(x)
            x = self.last_stage_conv2(x)
            x = return_activation(x, 'HS')
            return x


class LiteRASSP(Layer):
    def __init__(self, shape=(224, 224), n_class=2, avg_pool_kernel=(49, 49),
                 avg_pool_strides=(16, 20),
                 resize_method=ResizeMethod.BILINEAR):
        """LiteRASSP.
        # Arguments
            # init
                input_shape: Tuple/list of 2 integers, spatial shape of input
                    tensor
                n_class: Integer, number of classes.
                avg_pool_kernel: Tuple/integer, size of the kernel for
                    AveragePooling
                avg_pool_strides: Tuple/integer, stride for applying the of
                    AveragePooling operation
            # Call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            Output tensor of the original shape
            """
        super(LiteRASSP, self).__init__()
        self.shape = shape
        self.n_class = n_class
        self.avg_pool_kernel = avg_pool_kernel  # 11
        self.avg_pool_strides = avg_pool_strides  # 4
        self.resize_method = resize_method
        # branch1
        self.branch1_convblock = ConvBlock(128, 1, strides=1, nl='RE')
        # branch2
        self.branch2_avgpool = AveragePooling2D(pool_size=self.avg_pool_kernel,
                                                strides=self.avg_pool_strides)
        self.branch2_conv = Conv2D(128, 1, strides=1)
        # bracnh3
        self.branch3_conv = Conv2D(self.n_class, 1, strides=1)
        # merge1_2
        self.merge1_2_conv = Conv2D(self.n_class, 1, strides=1)

    def call(self, inputs, training=True):
        out_feature8, out_feature16 = inputs
        # branch1
        x1 = self.branch1_convblock(out_feature16, training=training)
        # branch2
        s = x1.shape
        x2 = self.branch2_avgpool(out_feature16)
        x2 = self.branch2_conv(x2)
        x2 = Activation('sigmoid')(x2)
        x2 = tf.image.resize(x2,
                             size=(int(s[1]), int(s[2])),
                             method=self.resize_method,
                             preserve_aspect_ratio=False,
                             antialias=False,
                             name=None)
        # branch3
        x3 = self.branch3_conv(out_feature8)
        # merge1_2
        x = Multiply()([x1, x2])
        x = tf.image.resize(x,
                            size=(int(2*s[1]), int(2*s[2])),
                            method=self.resize_method,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            name=None)
        x = self.merge1_2_conv(x)
        # merge3
        x = Add()([x, x3])
        # # out
        x = tf.image.resize(x,
                            size=self.shape,
                            method=self.resize_method,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            name=None)
        x = Activation('sigmoid')(x)
        # x = tf.nn.softmax(x, axis=-1)
        return x


class MobileNetV3SmallSegmentation(Model):
    def __init__(self, alpha=1.0, shape=(224, 224), n_class=2,
                 avg_pool_kernel=(11, 11), avg_pool_strides=(4, 4),
                 resize_method=ResizeMethod.BILINEAR, backbone='small'):
        """MobileNetV3SmallSegmentation.
        # Arguments
            # init
                alpha: Integer, width multiplier.
                input_shape: Tuple/list of 2 integers, spatial shape of input
                    tensor
                n_class: Integer, number of classes.
                avg_pool_kernel: Tuple/integer, size of the kernel for
                    AveragePooling
                avg_pool_strides: Tuple/integer, stride for applying the of
                    AveragePooling operation
                resize_method: Object, One from tensorflow.image.ResizeMethod
                backbone: String, name of backbone to use
            # Call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            Result of segmentation
            """
        super(MobileNetV3SmallSegmentation, self).__init__()
        if backbone == 'small':
            self.backbone = MobileNetV3SmallBackbone(
                alpha=alpha, mode='segmentation')
        self.segmentation_head = LiteRASSP(shape=shape,
                                           n_class=n_class,
                                           avg_pool_kernel=avg_pool_kernel,
                                           avg_pool_strides=avg_pool_strides,
                                           resize_method=resize_method)

    def call(self, inputs, training=True):
        segm_inputs = self.backbone(inputs, training)
        output = self.segmentation_head(segm_inputs, training)
        return output
