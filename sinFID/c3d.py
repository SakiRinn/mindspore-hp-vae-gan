import mindspore.nn as nn
import mindspore.ops as ops
import mindspore_hub as mshub


class C3D(nn.Cell):
    """Pretrained C3D network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,
        128: 1,
        256: 2,
        512: 3
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 is_training=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        is_training : bool
            If true, use train mode.
        """
        super(C3D, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 4, \
            'Last possible output block index is 4'

        self.blocks = nn.CellList([])

        model = "mindspore/1.9/inceptionv3_imagenet2012"
        c3d = mshub.load(model, num_classes=1000)

        block0 = nn.SequentialCell([
            c3d.conv1,
            ])
        self.blocks.append(block0)

        if self.last_needed_block >= 1:
            block1 = nn.SequentialCell([
                c3d.pool1,
                c3d.conv2,
            ])
            self.blocks.append(block1)

        if self.last_needed_block >= 2:
            block2 = nn.SequentialCell([
                c3d.pool2,
                c3d.conv3a,
                c3d.conv3b,
            ])
            self.blocks.append(block2)

        if self.last_needed_block >= 3:
            block3 = nn.SequentialCell([
                c3d.pool3,
                c3d.conv4a,
                c3d.conv4b,
            ])
            self.blocks.append(block3)

        if self.last_needed_block >= 4:
            block4 = nn.SequentialCell([
                c3d.pool4,
                c3d.conv5a,
                c3d.conv5b,
            ])
            self.blocks.append(block4)

        if self.last_needed_block >= 5:
            block5 = nn.SequentialCell([
                c3d.pool5,
            ])
            self.blocks.append(block5)

        self.set_train(is_training)

    def construct(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : mindspore.Tensor
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of mindspore.Tensor, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = ops.ResizeBilinear((299, 299), align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            if idx == 5:
                x = x.view(-1, 512 * 2, 7, 7)
                x = self.pad(x)
                x = x.view(-1, 512, 2, 8, 8)

            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
