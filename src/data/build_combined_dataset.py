import os
import pandas as pd


def process_and_merge_features(
    labels_path,
    eunice_path,
    nomin_path,
    prithvi_path,
    jade_path,
    output_path
):

    print("=" * 50)
    print("Loading Files...")
    print("=" * 50)

    labels = pd.read_csv(labels_path)
    e = pd.read_csv(eunice_path)
    n = pd.read_csv(nomin_path)
    p = pd.read_csv(prithvi_path)
    j = pd.read_csv(jade_path)

    print("Labels:", labels.shape)
    print("Eunice:", e.shape)
    print("Nomin:", n.shape)
    print("Prithvi:", p.shape)
    print("Jade:", j.shape)

    print("\nMerging datasets...")

    merged = (
        labels
        .merge(e, on="id", how="inner")
        .merge(n, on="id", how="inner")
        .merge(p, on="id", how="inner")
        .merge(j, on="id", how="inner")
    )

    print("Final merged shape:", merged.shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    merged.to_csv(output_path, index=False, compression="gzip")

    print("Saved:", output_path)


def main():

    print("\nBuilding combined datasets...\n")

    train_path = "src/feature_inputs/train/"
    test_path = "src/feature_inputs/test/"
    data_path = "src/data/"

    # TRAIN DATASET
    process_and_merge_features(
        labels_path=train_path + "labels_train.csv",
        eunice_path=train_path + "eunice_train_all_features.csv.gz",
        nomin_path=train_path + "nomin_combined_train_n.csv.gz",
        prithvi_path=train_path + "prithvi_train_2.csv.gz",
        jade_path=train_path + "jade_train_features.csv",
        output_path=data_path + "combined_train_with_labels.csv.gz"
    )

    # TEST DATASET
    process_and_merge_features(
        labels_path=test_path + "labels_test.csv",
        eunice_path=test_path + "eunice_test_all_features.csv.gz",
        nomin_path=test_path + "nomin_combined_test_n.csv.gz",
        prithvi_path=test_path + "prithvi_test_2.csv.gz",
        jade_path=test_path + "jade_test_features.csv",
        output_path=data_path + "combined_test_with_labels.csv.gz"
    )

    print("\nDone building datasets.")


if __name__ == "__main__":
    main()