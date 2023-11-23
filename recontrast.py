import torch
import torch.nn as nn


class ReContrast(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_freeze,
        bottleneck,
        decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        with torch.no_grad():
            en = self.encoder(x)
            return en[2]

        # en = self.encoder(
        #     x
        # )  # if input_size=256, then [[bs, 256, 64, 64], [bs, 512, 32, 32], [bs, 1024, 16, 16]]
        # with torch.no_grad():
        #     en_freeze = self.encoder_freeze(x)

        # en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        # de = self.decoder(self.bottleneck(en_2))
        # de = [a.chunk(dim=0, chunks=2) for a in de]
        # de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        # return (
        #     en_freeze + en,
        #     de,
        # )  # de's first half is recons of en, second half is recons of en_freeze
