import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



def get_model_FineTuning_classSize():
    """
    1. Pre-trained 모델을 이용한 Fine tuning 실시
    ===============================================================================
    a. num_classes: 모델이 구별할 Class의 수(txt, 배경)

    b. in_features: ROI head의 classifier가 입력받는 Feature의 차원 크기
       >>> 분류기(classifier)로 들어가는 입력 벡터의 길이
       >>> 모델이 Back bone network로부터 추출한 Feature vector의 크기

    c. FastRCNNPredictor(in_features, num_classes)
       >>> Standard classification + bounding box regression layers for Fast R-CNN.
       >>> in_channelrs: number of input channels
       >>> num_classes: number of output classes(including background)
       >>> head 내부의 box predictor를 새로운 것으로 교체하는 과정으로 미리 학습된 모델을
           사용자의 특정 Task에 맞게 조정하는 Fine-tuning 과정의 하나.
       >>> (cls_score): class score, 각 ROI가 특정 클래스(사람, 배경 등)에 속할 확률 계산.
           'Linear(in_features=1024, out_features=2, bias=True)'의 선형 Layer는 각
           잠재 영역에 대한 Class 확률 예측
       >>> (bbox_pred): 바운딩 박스 예측(Bounding box predictor)은 각 ROI에 대한 Bounding
           box의 위치 조정.
           'Linear(in_features=1024, out_features=8, bias=True)'에서 out_feautres=8은
           각 Class 별로 4개의 좌표(x1, y1, x2, y2)를 예측하기 위한 값으로, 두 클래스 각각에
           대해 4개씩 값을 예측하기 위해 총 8개의 출력이 필요함.
    ===============================================================================
    """
    # A. Box predictor를 본 Task에 맞게 조정
    ##################################################################################
    # A-1. Get model: COCO로 pre-train된 모델을 가지고 온다.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # A-2. 사용자의 Task에 맞게 Box predictor 조정
    num_classes = 2  # 대상 class(Text) + 배경 = 2
    # 초기 model(fine tuning 전)의 입력 Feature vector의 크기 조회.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Faster R-CNN 의 ROI Head 내부의 박스 예측기(Box predictor)를 새로운 것으로 교체
    # 미리 학습된 모델을 사용자의 특정 작업에 맞게 조정하는 Fine-tuning의 일부
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ##################################################################################

    return model



def construct_my_Faster_RCNN_model():

    """
    2. torchvision을 이용한 Faster R-CNN 모델 구성.
    ===============================================================================
    a. Backbone network 설정
       - 해당 코드에선 torchvision.models.mobilnet_v2을 백본 네트워크로 사용.
         > 이미지에서 Feature를 추출하는 역할
       - backbone.out_channels = 1280
         > 백본 네트워크의 출력 채널 수 설정.
         > Backbone 모델의 최종 출력 채널의 크기를 확인하고 해당 값을 지정해주자.
         > Faster R-CNN 모델은 이 정보를 알아야만 RPN과 ROI Pooling이 올바르게 작동 가능.

    b. Anchor Generator 정의
       - AnchorGenerator는 RPN이 사용할 Anchor를 생성함.
       - Anchor는 다양한 크기('sizes')와 종횡비('aspect_rations')의 조합을 사용하여 다양한
         형태와 크기의 객체를 탐지할 수 있도록 함.

    c. ROI Pooler 설정
       - 'MultiScaleRoIAlign'는 RPN에서 제안한 후보 영역(Region proposal)에 대해 Feature를
         잘라내고 재조정하는 작업 수행.
         > 'output_size': 재조정된 Feature의 크기
         > 'sampling_ratio': 샘플링 비율

    d. Faster R-CNN 모델 구성
       - 'FasterRCNN' 클래스를 사용해 백본, 앵커 생성기, ROI 풀러 등을 결합해 전체 Faster
         R-CNN 모델을 구성.
         > 'num_classes=2'는 모델이 탐지해야할 클래스의 수 임.
         > 객체와 배경 두 가지 클래스 가정.
    ===============================================================================
    """

    # a. Backbone network 설정
    ###############################################################################
    # 분류 목적으로 미리 학습된 모델을 로그하고 Features만을 Return하자.
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # Faster RCNN은 백본의 출력 채널 수를 알아야 합니다.
    # mobilenetV2의 경우 1280이므로 여기에 추가해야 합니다.
    # 해당 작업은 - model의 구조를 출력하고, 그 구조의 가장 마지막 레이어의 출력차원의 크기를 수작업으로 봐야하는듯 하다.
    backbone.out_channels = 1280
    ###############################################################################

    # b. Anchor Generator 정의
    ###############################################################################
    # RPN(Region Proposal Network)이 5개의 서로 다른 크기와 3개의 다른 측면 비율(Aspect ratio)을 가진 5 x 3개의 앵커를 공간 위치마다 생성하도록 함.
    # 각 Feature map이 잠재적으로 다른 사이즈와 측면 비율을 가질 수 있기 때문에 Tuple[Tuple[int]] 타입을 가지도록 합니다.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    ###############################################################################

    # c. ROI Pooler 설정
    ###############################################################################
    # 관심 영역의 자르기 및 재할당 후 자르기 크기를 수행하는 데 사용할 피쳐 맵을 정의합니다.
    # 만약 백본이 텐서를 리턴할때, featmap_names 는 [0] 이 될 것이라고 예상합니다.
    # 일반적으로 백본은 OrderedDict[Tensor] 타입을 리턴해야 합니다.
    # 그리고 특징맵에서 사용할 featmap_names 값을 정할 수 있습니다.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    ###############################################################################

    # d. Faster R-CNN 모델 구성
    ###############################################################################
    # 조각들을 Faster RCNN 모델로 합칩니다.
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    ###############################################################################

    return model