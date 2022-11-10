import mindspore.nn as nn
import mindspore.ops as ops
import mindspore_hub as mshub


class InceptionV3(nn.Cell):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
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
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.CellList([])

        model = "mindspore/1.9/inceptionv3_imagenet2012"
        inception = inception = mshub.load(model, num_classes=1000)

        # Block 0: input to maxpool1
        block0 = nn.SequentialCell([
            inception.Conv2d_1a,
            inception.Conv2d_2a,
            inception.Conv2d_2b,
            ])
        self.blocks.append(block0)

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = nn.SequentialCell([
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b,
                inception.Conv2d_4a,
            ])
            self.blocks.append(block1)

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = nn.SequentialCell([
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ])
            self.blocks.append(block2)

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = nn.SequentialCell([
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ])
            self.blocks.append(block3)

        if self.last_needed_block >= 4:
            block4 = nn.SequentialCell([
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ])
            self.blocks.append(block4)

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
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
