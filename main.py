import argparse
import logging
from src.pipeline import DriftDetectorPipeline
from src.utils import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description='Drift Detector')
    parser.add_argument('--data-size', type=int, default=1000, help='Size of the synthetic dataset')
    parser.add_argument('--drift-size', type=int, default=500, help='Size of the drift in the dataset')
    parser.add_argument('--window-size', type=int, default=200, help='Size of the window for drift detection')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data = generate_synthetic_data(args.data_size, args.drift_size)
    pipeline = DriftDetectorPipeline(args.window_size)
    pipeline.fit(data)
    pipeline.predict(data)

if __name__ == '__main__':
    main()