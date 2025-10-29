import argparse

import mtcnn.train.gen_pnet_train as gptd
import mtcnn.train.gen_landmark as gl
from mtcnn.datasets import get_by_name


parser = argparse.ArgumentParser(
    description='Generate training data for pnet.')
parser.add_argument('-o', dest="output_folder", default="output/data_train", type=str, help="Folder to save training data for pnet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")
args = parser.parse_args()

detection_dataset = get_by_name(args.detection_dataset)
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
gptd.generate_training_data_for_pnet(detection_meta, output_folder=args.output_folder, suffix='pnet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
gptd.generate_training_data_for_pnet(detection_eval_meta, output_folder=args.output_folder, suffix='pnet_eval')
print("Done")
