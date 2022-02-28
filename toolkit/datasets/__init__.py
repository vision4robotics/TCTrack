from .uav import UAVDataset
from .dtb import DTBDataset
from .uavdt import UAVDTDataset
from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .visdrone import VISDRONEDDataset
from .visdrone1 import VISDRONED2018Dataset
from .uavtrack112 import UAVTrack112Dataset
from .uavtrack112_l import UAVTrack112lDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'DTB' in name:
            dataset = DTBDataset(**kwargs)
        elif 'UAV10' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'UAVTrack112_l' in name:
            dataset = UAVTrack112lDataset(**kwargs)
        elif 'VISDRONED2019' in name:
            dataset = VISDRONEDDataset(**kwargs)
        elif 'UAVDARK' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'VISDRONED2018' in name:
            dataset = VISDRONED2018Dataset(**kwargs)
        elif 'UAVTrack' in name:
            dataset = UAVTrack112Dataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV123' in name:
            dataset = UAVDataset(**kwargs)
        elif 'V4R' in name:
            dataset = V4RDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

