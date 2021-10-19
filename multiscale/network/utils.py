
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn

#from network.mynn import Norm2d, Upsample
import network.hrnetv2 as hrnetv2

from config import cfg

def get_trunk(trunk_name, output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, 'Only stride8 supported right now'
    
    if trunk_name == 'hrnetv2':
        backbone = hrnetv2.get_seg_model()
        high_level_ch = backbone.high_level_ch
        s2_ch = -1
        s4_ch = -1
    else:
        raise 'unknown backbone {}'.format(trunk_name)

    return backbone, s2_ch, s4_ch, high_level_ch


def BNReLU(ch):
    return nn.Sequential(
        nn.BatchNorm2d(ch),
        nn.ReLU())


def init_attn(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.5)
        elif isinstance(module, cfg.MODEL.BNFUNC):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


def make_attn_head(in_ch, out_ch):
    bot_ch = cfg.MODEL.SEGATTN_BOT_CH
    if cfg.MODEL.MSCALE_OLDARCH:
        return old_make_attn_head(in_ch, bot_ch, out_ch)

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', nn.BatchNorm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    if cfg.MODEL.MSCALE_INNER_3x3:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = nn.BatchNorm2d(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    if cfg.MODEL.MSCALE_DROPOUT:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


def old_make_attn_head(in_ch, bot_ch, out_ch):
    attn = nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=out_ch, bias=False),
        nn.Sigmoid())

    init_attn(attn)
    return attn
