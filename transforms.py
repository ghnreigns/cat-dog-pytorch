import albumentations
import cv2


class Augmentation(ABC):
    @abstractmethod
    def augment(image):
        """Augment an image."""


class AlbumentationsAugmentation(Augmentation):
    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        albu_dict = {"image": image}
        transform = self.transforms(**albu_dict)
        return transform["image"]


def get_albu_transforms(config, probs: float = 1.0):
    transforms_train = albumentations.Compose(
        [
            # albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(
                scale_limit=0.20,
                rotate_limit=10,
                shift_limit=0.1,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            # albumentations.VerticalFlip(p=0.66),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Cutout(num_holes=8, p=1),
            albumentations.Resize(
                height=config.image_size, width=config.image_size, p=1.0
            ),
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    transforms_test = albumentations.Compose(
        [
            albumentations.Resize(
                height=config.image_size, width=config.image_size, p=1.0
            ),
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    return transforms_train, transforms_test


def get_albu_dict(positional_augs: List, color_augs: List):

    positional_augs = [
        "VerticalFlip",
        "HorizontalFlip",
        "RandomResizedCrop",
        "ShiftScaleRotate",
    ]
    color_augs = ["HueSaturationValue", "RandomBrightnessContrast"]
    all_augs = positional_augs + color_augs
    aug_dict = {}
    for aug_name in all_augs:
        if aug_name not in aug_dict:
            aug_dict[aug_name] = generate_default_configuration(
                lambda item_name: getattr(albumentations, item_name), item_name=aug_name
            )
            # set p = 1 to ensure augs go through
            aug_dict[aug_name]["p"] = 1
    transforms_dict = {}
    for aug_keys, aug_values in aug_dict.items():
        if "Resized" in aug_keys:
            # means got default positional arguments height and width
            continue
        transforms_dict[aug_keys] = albumentations.Compose(
            [
                getattr(albumentations, aug_keys)(**aug_values),
                albumentations.Resize(
                    height=config.image_size, width=config.image_size, p=1.0
                ),
                albumentations.Normalize(),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )