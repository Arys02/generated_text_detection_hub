#src/detector/detector_factory.py
from src.detector.cristal_anchor.cristal_anchor import CristalAnchor
from src.detector.detect_gpt import DetectGPT
from src.detector.fast_detect_gpt import FastDetectGPT
from src.detector.random_base_detector import RandomBaseDetector
from src.detector.binocular import BinocularDetector
from src.detector.ada_detect_gpt.ada_detect_gpt import AdaDetectGPT
from src.detector.subjectivity_aware_generation_estimator.sage import Sage


def detector_factory(config):
    name = config['name']

    if name == 'dummy':
        detector = RandomBaseDetector(config)
    elif name == 'binocular':
        detector = BinocularDetector(config)
    elif name == 'detect_gpt':
        detector = DetectGPT(config)
    elif name == 'fast_detect_gpt':
        detector = FastDetectGPT(config)
    elif name == 'ada_detect_gpt':
        detector = AdaDetectGPT(config)
    elif name == 'cristal_anchor':
        detector = CristalAnchor(config)
    elif name == 'sage':
        detector = Sage(config)
    else:
        raise ValueError('Unknown detector {}'.format(name))
    return detector
