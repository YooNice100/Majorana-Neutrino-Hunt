import os
import pandas as pd


def process_and_merge_features(
    eunice_path,
    nomin_path,
    prithvi_path,
    jade_path,
    output_path
):

    print("=" * 60)
    print("Loading feature files...")
    print("=" * 60)

    eunice = pd.read_csv(eunice_path)
    nomin = pd.read_csv(nomin_path)   # contains labels
    prithvi = pd.read_csv(prithvi_path)
    jade = pd.read_csv(jade_path)

    print("Eunice:", eunice.shape)
    print("Nomin:", nomin.shape)
    print("Prithvi:", prithvi.shape)
    print("Jade:", jade.shape)

    print("\nMerging datasets...")

    merged = (
        nomin
        .merge(eunice, on="id", how="inner")
        .merge(prithvi, on="id", how="inner")
        .merge(jade, on="id", how="inner")
    )

    print("Final merged shape:", merged.shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    merged.to_csv(output_path, index=False, compression="gzip")

    print("Saved:", output_path)

    return merged


def main():

    print("\nBuilding combined datasets...\n")

    # TRAIN
    process_and_merge_features(
        eunice_path="feature_inputs/eunice_train_all_features.csv.gz",
        nomin_path="feature_inputs/nomin_combined_train_n.csv.gz",
        prithvi_path="feature_inputs/prithvi_train_2.csv.gz",
        jade_path="feature_inputs/jade_train_features.csv",
        output_path="data/combined_train_with_labels.csv.gz"
    )

    # TEST
    process_and_merge_features(
        eunice_path="feature_inputs/eunice_test_all_features.csv.gz",
        nomin_path="feature_inputs/nomin_combined_test_n.csv.gz",
        prithvi_path="feature_inputs/prithvi_test_2.csv.gz",
        jade_path="feature_inputs/jade_test_features.csv",
        output_path="data/combined_test_with_labels.csv.gz"
    )

    print("\nDataset build complete.")


if __name__ == "__main__":
    main()