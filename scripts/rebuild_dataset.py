import argparse
import sys
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "datasets" / "raw"
    output_dir = base_dir / "datasets" / "verified_parquet_images"
    
    train_parquet_path = raw_dir / "train-00000-of-00001.parquet"
    test_parquet_path = raw_dir / "test-00000-of-00001.parquet"

    print("=" * 60)
    print("STEP 2: Verify Parquet Source")
    print("=" * 60)
    
    if not train_parquet_path.exists() or not test_parquet_path.exists():
        print("ERROR: Parquet files not found.")
        sys.exit(1)

    df_train = pd.read_parquet(train_parquet_path)
    df_test = pd.read_parquet(test_parquet_path)

    print(f"Train Parquet Row Count: {len(df_train)}")
    print(f"Test Parquet Row Count: {len(df_test)}")
    
    if "label" not in df_train.columns or "label" not in df_test.columns:
        print("ERROR: 'label' column missing.")
        sys.exit(1)

    train_labels = df_train['label'].value_counts().to_dict()
    test_labels = df_test['label'].value_counts().to_dict()
    
    print(f"Train labels: REAL(0)={train_labels.get(0, 0)}, FAKE(1)={train_labels.get(1, 0)}")
    print(f"Test labels:  REAL(0)={test_labels.get(0, 0)},  FAKE(1)={test_labels.get(1, 0)}")

    if len(df_train) + len(df_test) != 972:
        print("ERROR: Total rows != 972 as expected.")
        sys.exit(1)
        
    for split_labels in (train_labels, test_labels):
        for label in split_labels.keys():
            if label not in (0, 1):
                print(f"ERROR: Unexpected label found: {label}")
                sys.exit(1)

    print("\nParquet verification passed. Proceeding to Step 3: Rebuild.")
    
    print("=" * 60)
    print("STEP 3: Rebuild into verified_parquet_images")
    print("=" * 60)
    
    def extract(parquet_path, split_name):
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        
        real_dir = output_dir / split_name / "REAL"
        fake_dir = output_dir / split_name / "FAKE"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        real_count = 0
        fake_count = 0
        
        manifest_records = []

        print(f"Extracting {split_name}...")
        for i, row in enumerate(dataset):
            image = row["image"]
            label = row["label"]
            
            if label == 0:
                filename = f"real_{real_count:06d}.png"
                output_path = real_dir / filename
                image.convert("RGB").save(output_path)
                real_count += 1
                class_name = "REAL"
            elif label == 1:
                filename = f"fake_{fake_count:06d}.png"
                output_path = fake_dir / filename
                image.convert("RGB").save(output_path)
                fake_count += 1
                class_name = "FAKE"
                
            manifest_records.append({
                "source_index": i,
                "split": split_name,
                "class_name": class_name,
                "label": label,
                "filename": filename,
                "relative_path": str(output_path.relative_to(base_dir)).replace("\\", "/")
            })
            
        return manifest_records, real_count, fake_count

    train_records, train_real, train_fake = extract(train_parquet_path, "train")
    test_records, test_real, test_fake = extract(test_parquet_path, "test")
    
    manifest_df = pd.DataFrame(train_records + test_records)
    manifest_dir = base_dir / "datasets" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "verified_extraction_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print("\nExtraction complete.")
    print(f"Verified dataset saved to: {output_dir}")
    print(f"Extraction manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
