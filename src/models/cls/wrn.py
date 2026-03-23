from pytorchcv.model_provider import get_model


def wrn20_10_cifar10(num_classes=10, pretrained=False):
    return get_model("wrn20_10_cifar10", pretrained=pretrained)


def wrn20_10_cifar100(num_classes=100, pretrained=False):
    return get_model("wrn20_10_cifar100", pretrained=pretrained)


def wrn20_10_svhn(num_classes=10, pretrained=False):
    return get_model("wrn20_10_svhn", pretrained=pretrained)
