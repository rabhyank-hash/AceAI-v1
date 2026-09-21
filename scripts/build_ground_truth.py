"""Write data/processed/ground_truth/<course>.json for every course in data/raw/."""

from aceai.config import DATA_PROCESSED, DATA_RAW
from aceai.ingest import load_all
from aceai.ingest.ground_truth import extract_ground_truth


def main() -> None:
    out_dir = DATA_PROCESSED / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    for course, los in load_all(DATA_RAW).items():
        gt = extract_ground_truth(course, los)
        path = out_dir / f"{course}.json"
        path.write_text(gt.model_dump_json(indent=2) + "\n", encoding="utf-8")
        n_mod = sum(len(u.modules) for u in gt.units)
        n_broad = sum(len(v) for v in gt.broad_lo_ids.values())
        print(
            f"{course}: {len(gt.units)} units, {n_mod} modules, "
            f"{len(gt.all_lo_ids()) - n_broad} LOs + {n_broad} broad -> {path}"
        )


if __name__ == "__main__":
    main()
