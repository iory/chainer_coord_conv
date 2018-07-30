import chainer
import chainer.functions as F
import chainer.links as L


class AddCoords(chainer.Chain):

    """
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def __call__(self, x):
        """
        Parameters
        ----------
        x : chainer.Variable
            shape(batch_size, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = x.shape
        xp = self.xp

        xx_channel = xp.tile(xp.arange(x_dim), (1, y_dim, 1))
        yy_channel = xp.tile(xp.arange(y_dim),  (1, x_dim, 1)).transpose(0, 2, 1)

        xx_channel = xp.array(xx_channel, 'f') / (x_dim - 1)
        yy_channel = xp.array(yy_channel, 'f') / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xp.tile(xx_channel, (batch_size, 1, 1, 1)).transpose(0, 1, 3, 2)
        yy_channel = xp.tile(yy_channel, (batch_size, 1, 1, 1)).transpose(0, 1, 3, 2)

        ret = F.concat([x, xx_channel, yy_channel], axis=1)

        if self.with_r:
            rr = F.sqrt(F.square(xx_channel - 0.5) + F.square(yy_channel - 0.5))
            ret = F.concat([ret, rr], axis=1)
        return ret


class CoordConv(chainer.Chain):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        with self.init_scope():
            self.addcoords = AddCoords(with_r=with_r)
            self.conv = L.Convolution2D(
                in_size, out_channels, 1, **kwargs)

    def __call__(self, x):
        h = self.addcoords(x)
        h = self.conv(h)
        return h
