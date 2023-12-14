import argparse
import os
# os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/" # communication
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/swordfish-datastore/wl2787/huggingface_cache/" # branzino
os.environ["TRANSFORMERS_CACHE"] = "/local/data/wl2787/huggingface_cache/" # coffee
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
import yaml
from munch import Munch
from utils.backbones import BackboneModel


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="conf/backbone.yaml")

parser.add_argument("--shots", type=str)
parser.add_argument("--source", type=str, default="ja")
parser.add_argument("--target", type=str, default="en")
parser.add_argument("--save_strategy", type=str, default="no")
parser.add_argument("--epochs", type=int, default=3)
cmd_args = vars(parser.parse_args())


# Load YAML config file
args = Munch.fromYAML(open(cmd_args["config_file"], "r"))
model_name = f"{args.model.checkpoint.split('/')[-1]}-language={args.model.language}-epochs={args.training.epochs}-batch_size={args.training.train_batch_size}"

if args.model.arguments != None:
    model_name += f"-arguments={args.model.arguments}"

# Create model directory
out_dir = os.path.join(args.training.save_checkpoint_dir, model_name)
args.model_name = model_name
args.out_dir = out_dir
args.model.tokenizer = args.model.checkpoint

print(yaml.dump(args, allow_unicode=True, default_flow_style=False))

train_data_dir = args.training.train_data_dir
val_data_dir = args.training.val_data_dir

model = BackboneModel(args)
model.train(train_data_dir, val_data_dir)


if cmd_args["shots"] != None:
    assert args.model.language == cmd_args["source"]

    model_name = f"{args.model.checkpoint.split('/')[-1]}-shots={cmd_args['shots']}"

    # Create model directory
    out_dir = os.path.join(args.training.save_checkpoint_dir, model_name)
    args.model_name = model_name
    args.out_dir = out_dir
    args.training.save_strategy = cmd_args["save_strategy"]
    args.training.epochs = cmd_args["epochs"]

    train_data_dir = f"data/training_lang={cmd_args['target']}-data=split_{cmd_args['target']}-shots={cmd_args['shots']}.csv"
    val_data_dir = f"data/val_lang={cmd_args['target']}-data=split_{cmd_args['target']}.csv"

    model.update_parameters(args)
    model.train(train_data_dir, val_data_dir)