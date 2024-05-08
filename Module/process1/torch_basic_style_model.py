from Module.Global_variable import torch, warnings

import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# retinaNet을 위한 Library
from functools import partial    # 함수의 일부 인자를 고정한 새로운 함수를 생성하는 데 사용.
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead




class get_my_torch_model:
    
    def __init__(
        self,
        class_num,
        model_key="faster_fpn",
        cnn_key="mobilenet_v3"
    ):
        self.class_num = class_num
        self.model_key = model_key
        self.cnn_key = cnn_key
        
        self.anchor_size = (32, 64, 128, 256, 512)
        self.anchor_ratio = (0.5, 1.0, 2.0)
        
        
    
    def process(self):
        if self.model_key == "faster_fpn":
            model = self.faster_RCNN_fpn_class_num_fine_tuning()
        elif self.model_key == "retinaNet":
            model = self.get_fine_tuned_retinaNet()
            
        else:
            model = torchvision.models.mobilenet_v2(weights="DEFAULT").features
            model = self.custom_Faster_RCNN_model_maker(
                backbone_model=model,
                backbone_out_channel_size=1280
            )    
        return model
    
    
    
    def faster_RCNN_fpn_class_num_fine_tuning(self):
        
        if self.cnn_key=="resnet50":
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif self.cnn_key=="resnet50_v2":
            model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        elif self.cnn_key=="mobilenet_v3_320":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        
        # FIne tuning 전 입력 Feature vector의 크기
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # ROI Header 내부의 Box predictor의 class size 수정
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.class_num)
        
        return model
    
    
    
    def get_fine_tuned_retinaNet(self):

        if self.cnn_key == "resnet50_v2":
            model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
        else:
            model = retinanet_resnet50_fpn(weights="DEFAULT")

        # number of classes를 수정한 classification head를 추가한다.
        # pre-train된 retina_model의 anchor 수
        number_of_anchors = model.head.classification_head.num_anchors
        # classification head 생성 및 덮어씌우기
        retina_new_classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=number_of_anchors,
            num_classes=self.class_num,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        model.head.classification_head = retina_new_classification_head

        return model
        
    
    
    def custom_Faster_RCNN_model_maker(self, backbone_model, backbone_out_channel_size):
        # Anchor Generator 정의
        anchor_generator = AnchorGenerator(
            sizes=(anchor_size,), aspect_ratios=(anchor_ratio,)
        )
        # ROI Pooler 정의
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        # Faster R-CNN 모델 구성
        model = FasterRCNN(
            backbone=backbone_model,
            num_classes=self.class_num,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        return model
        
        
        
        
def get_optimizer(model, learning_rate=0.0001, weight_decay=0.0005, opt_key="Adam"):
    """
    선택 가능한 Optimizer는 Adam, AdamW, AdaGrad, RMSprop, SGD이며, Default는 Adam이다.
    -----------------------------------------------------------------------------------
    조정 가능한 Hyper Parameter는 learning_rate와 weight_decay 두가지다.
    """
    params = model.parameters()
    # 학습이 필요한 Parameter들만 선택받는다.
#     params = [p for p in model.parameters() if p.requires_grad]

    if opt_key == "Adam":
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_key == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_key == "AdaGrad":
        optimizer = torch.optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_key == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_key == "SGD":
        optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        warnings.warn(
            """
            You must choose a proper optimizer. The default optimizer Adam will be automatically selected.
            You can choose Adam, AdamW, AdaGrad, RMSprop, or SGD.
            """
        )
        optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        
    return optimizer