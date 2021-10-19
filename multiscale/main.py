from network.ocrnet import HRNet
from loss.rmi import RMILoss

hrnet = HRNet(2, RMILoss(
            num_classes=2,
            ignore_index=255))