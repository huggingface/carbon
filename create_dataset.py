from datasets import load_dataset

DATASET_ID = "TIGER-Lab/MMLU-Pro"
OUT_ID = "hf-carbon/mmlu-pro-biology"


def main() -> None:
    # Load dataset from the Hub
    mmlu = load_dataset(DATASET_ID)

    # Filter for biology category
    filtered = mmlu.filter(lambda ex: ex.get("category") == "biology")

    # Push to Hub
    filtered.push_to_hub(OUT_ID)

    print("Pushed", OUT_ID)
    for split, ds in filtered.items():
        print(split, ds.num_rows)


if __name__ == "__main__":
    main()
