# 模型包初始化文件
from .cosface import get_id_extractor, IDExtractor
from .dil.dil_model import DIL
from .fal.fal_model import FALModel
from .svd import StableVideoDiffusion, get_svd_model

# 导入训练器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hifivfs_trainer import HiFiVFSTrainer

# 导出所有模块
__all__ = [
    'get_id_extractor', 'IDExtractor',
    'DIL', 'FALModel',
    'StableVideoDiffusion', 'get_svd_model',
    'HiFiVFSTrainer'
]
