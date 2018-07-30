import argparse

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from dataset import CoordConvDataset
from coord_conv import CoordConv


class Transform(object):

    def __init__(self):
        pass

    def __call__(self, in_data):
        x, y, _ = in_data
        x = x / (64.0 - 1.0)
        x = np.tile(np.array(x, 'f'), (1, 64, 64))
        y = np.argmax(y.reshape(-1))
        return x, y


class Model(chainer.Chain):

    def __init__(self, use_coordconv=True):
        super(Model, self).__init__()
        with self.init_scope():
            if use_coordconv:
                self.coord_conv1 = CoordConv(2, 32)
            else:
                self.coord_conv1 = L.Convolution2D(2, 32, 1)
            self.conv2 = L.Convolution2D(32, 32, 1)
            self.conv3 = L.Convolution2D(32, 1, 1)

    def __call__(self, x):
        batch_size = x.shape[0]
        h = self.coord_conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = h.reshape(batch_size, -1)
        return h


def main():
    parser = argparse.ArgumentParser(description='Train CoordConv')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--nouniform', action='store_true')
    parser.add_argument('--nocoordconv', action='store_true')
    args = parser.parse_args()

    model = L.Classifier(Model(use_coordconv=not args.nocoordconv))
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(0.01)
    optimizer.setup(model)

    type = 'uniform'
    if args.nouniform:
        type = 'quadrant'

    train = TransformDataset(
        CoordConvDataset(split='train', type=type),
        Transform())
    test = TransformDataset(
        CoordConvDataset(split='test', type=type),
        Transform())

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
