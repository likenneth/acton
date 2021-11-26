import numpy as np
import random
from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')

from plb.datamodules.seq_datamodule import SkeletonTransform


class TrainDataTransform(object):
    def __init__(self, aug_shift_prob, aug_shift_range, aug_rot_prob, aug_rot_range, min_length, max_length, aug_time_prob, aug_time_rate) -> None:
        self.train_transform = SkeletonTransform(aug_shift_prob, aug_shift_range, aug_rot_prob, aug_rot_range, min_length, max_length, aug_time_prob, aug_time_rate)
        self.min_length = min_length
        self.max_length = max_length
        self.aug_time_prob = aug_time_prob
        self.aug_time_rate = aug_time_rate

    def __call__(self, sample):
        transform = self.train_transform
        ttl = sample.size(0)

        # let's random crop
        len = random.randint(self.min_length, min(self.max_length, ttl))
        start = random.randint(0, ttl - len)
        sample = sample[start:start + len]

        xi, veloi = transform(sample)#, shut=True)  # not do transform on one branch
        xj, veloj = transform(sample)

        return xi, xj, veloi, veloj  # self.online_transform(sample)


class EvalDataTransform(TrainDataTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FinetuneTransform(object):
    def __init__(
            self,
            input_height: int = 224,
            jitter_strength: float = 1.,
            normalize=None,
            eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height)
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `GaussianBlur` from `cv2` which is not installed yet.')

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
