import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Remove the solver states stored in a trained model")
    parser.add_argument(
        "model",
        default="/home/stureski/pretrained-slow-fast/SLOWFAST_VGG.pyth",
        help="path to the input model file",
    )

    args = parser.parse_args()

    model = torch.load(args.model)
    del model["optimizer_state"]

    filename_wo_ext, ext = os.path.splitext(args.model)
    output_file = filename_wo_ext + "_wo_solver_states" + ext
    torch.save(model, output_file)
    print("Done. The model without solver states is saved to {}".format(output_file))

if __name__ == "__main__":
    main()
