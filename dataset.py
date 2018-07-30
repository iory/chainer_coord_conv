import numpy as np
from sklearn.model_selection import train_test_split
import chainer.functions as F
from chainer.dataset import DatasetMixin


class CoordConvDataset(DatasetMixin):

    def __init__(self, split='train', type='uniform'):
        super(CoordConvDataset, self).__init__()
        onehot = np.pad(np.eye(3136).reshape((3136, 56, 56, 1)),
                        ((0,0), (4,4), (4,4), (0,0)), "constant").transpose(0, 3, 1, 2)
        self.images = F.convolution_2d(
            onehot,
            np.ones((1, 1, 9, 9), 'f'), stride=1, pad=4).data

        self.onehot2 = onehot
        if type == 'uniform':
            indices = np.arange(0, len(onehot), dtype='int32')
            train, test = train_test_split(
                indices, test_size=0.2, random_state=0)
            if split == 'train':
                self.indices = train
            else:
                self.indices = test
            self.onehot = np.vstack(np.where(onehot[:, 0, ...]))[1:].T
            self.onehot = self.onehot.reshape(-1, 2, 1, 1)
        elif type == 'quadrant':
            # Create the quadrant datasets
            pos = np.where(onehot == 1.0)
            X = pos[3]
            Y = pos[2]

            train_ids = []
            test_ids = []

            for i, (x, y) in enumerate(zip(X, Y)):
                if x > 32 and y > 32:  # 4th quadrant
                    test_ids.append(i)
                else:
                    train_ids.append(i)

            self.onehot = np.vstack(np.where(onehot[:, 0, ...]))[1:].T
            self.onehot = self.onehot.reshape(-1, 2, 1, 1)
            if split == 'train':
                self.indices = np.array(train_ids, 'i')
            else:
                self.indices = np.array(test_ids, 'i')
        else:
            raise ValueError

    def __len__(self):
        return len(self.indices)

    def get_example(self, i):
        """
        Returns the i-th example.
        """
        return (self.onehot[self.indices[i]],
                self.onehot2[self.indices[i]],
                self.images[self.indices[i]])
