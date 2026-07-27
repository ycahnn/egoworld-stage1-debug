from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/hdd2/youngchan/seoin/egoworld-stage1-debug/data/preprocessed_224_png_nvme/combined_subjects1234")


def parse(path: str):
    parts = Path(path).parts
    return parts[0], parts[1], int(parts[2])


def main():
    rows = []
    for split in ("train", "val", "test"):
        csv = ROOT / f"h2o_subjects1234_{split}_pre224.csv"
        df = pd.read_csv(csv, usecols=["subject", "original_image_path"])
        parsed = df["original_image_path"].map(parse)
        df[["path_subject", "env", "clip"]] = pd.DataFrame(parsed.tolist(), index=df.index)
        grouped = df.groupby(["subject", "env", "clip"]).size().reset_index(name="rows")
        grouped["split"] = split
        rows.append(grouped)

    all_rows = pd.concat(rows, ignore_index=True)
    print("## rows by split")
    print(all_rows.groupby("split")["rows"].sum().to_string())

    print("\n## clip map")
    for subject in sorted(all_rows["subject"].unique()):
        print(f"\n{subject}")
        for env in sorted(all_rows.loc[all_rows["subject"] == subject, "env"].unique()):
            parts = []
            for split in ("train", "val", "test"):
                subset = all_rows[
                    (all_rows["subject"] == subject)
                    & (all_rows["env"] == env)
                    & (all_rows["split"] == split)
                ].sort_values("clip")
                clips = ",".join(map(str, subset["clip"].tolist())) or "-"
                frames = int(subset["rows"].sum()) if len(subset) else 0
                parts.append(f"{split}={clips} ({frames})")
            print(f"  {env}: " + " | ".join(parts))

    print("\n## per-subject totals")
    totals = all_rows.groupby(["subject", "split"])["rows"].sum().unstack(fill_value=0)
    print(totals.to_string())


if __name__ == "__main__":
    main()
