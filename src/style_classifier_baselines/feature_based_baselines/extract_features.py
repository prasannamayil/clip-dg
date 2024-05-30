import torch
import argparse
import utils
import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", required=True)
    parser.add_argument("--layer-names", "-l", nargs="+", required=True)
    parser.add_argument("--dataset-path", "-d", required=True)
    parser.add_argument("--output-path", "-o", required=True)
    parser.add_argument("--batch-size", "-b", default=256, type=int)

    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available else "cpu"

    transform, model, initialize_feature_extractor = utils.prepare_intermediate_features_model_and_transform(
        args.model_name, args.layer_names, device)
    data_loader = utils.get_data_loader(
        args.dataset_path, args.batch_size, transform, shuffle=False)

    results = {"labels": []}
    for ln in args.layer_names:
        results[ln] = []

    with torch.no_grad():
        with initialize_feature_extractor as feature_extractor:
            for x, y in tqdm.tqdm(data_loader):
                results["labels"].append(y)
                model_output = model(x.to(device))

                for ln in args.layer_names:
                    if ln == "output":
                        features = model_output
                    else:
                        features = feature_extractor(ln)
                    results[ln].append(features.cpu())

    for ln in args.layer_names:
        results[ln] = torch.cat(results[ln], 0)

    torch.save(results, args.output_path)


if __name__ == "__main__":
    main()
