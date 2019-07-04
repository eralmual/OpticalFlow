
class Config():

    def __init__(self):

        self.USE_FPN = False
        # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
        self.ANCHOR_SIZES = (32, 64, 128, 256, 512)
        # Stride of the feature map that RPN is attached.
        # For FPN, number of strides should match number of scales
        self.ANCHOR_STRIDE = (16,)
        # RPN anchor aspect ratios
        self.ASPECT_RATIOS = (0.5, 1.0, 2.0)
        # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
        # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
        self.STRADDLE_THRESH = 0
        # Minimum overlap required between an anchor and ground-truth box for the
        # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
        # ==> positive RPN example)
        self.FG_IOU_THRESHOLD = 0.7
        # Maximum overlap allowed between an anchor and ground-truth box for the
        # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
        # ==> negative RPN example)
        self.BG_IOU_THRESHOLD = 0.3
        # Total number of RPN examples per image
        self.BATCH_SIZE_PER_IMAGE = 256
        # Target fraction of foreground (positive) examples per RPN minibatch
        self.POSITIVE_FRACTION = 0.5

        # Number of top scoring RPN proposals to keep after combining proposals from
        # all FPN levels
        self.FPN_POST_NMS_TOP_N_TRAIN = 2000
        self.FPN_POST_NMS_TOP_N_TEST = 2000

        # Number of top scoring RPN proposals to keep before applying NMS
        # When FPN is used, this is *per FPN level* (not total)
        self.PRE_NMS_TOP_N_TRAIN = 12000
        self.PRE_NMS_TOP_N_TEST = 6000

        # Number of top scoring RPN proposals to keep after applying NMS
        self.POST_NMS_TOP_N_TRAIN = 2000
        self.POST_NMS_TOP_N_TEST = 1000

        # Apply the post NMS per batch (default) or per image during training
        # (default is True to be consistent with Detectron, see Issue #672)
        self.FPN_POST_NMS_PER_BATCH = True

        # NMS threshold used on RPN proposals
        self.NMS_THRESH = 0.7
        # Proposal height and width both need to be greater than RPN_MIN_SIZE
        # (a the scale used during training or inference)
        self.MIN_SIZE = 0