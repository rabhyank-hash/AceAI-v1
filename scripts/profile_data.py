"""Write data/processed/profile.md from every CSV in data/raw/."""

from aceai.config import DATA_PROCESSED, DATA_RAW
from aceai.ingest import load_all
from aceai.ingest.profile import render_profile


def main() -> None:
    by_course = load_all(DATA_RAW)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "profile.md"
    out.write_text(render_profile(by_course), encoding="utf-8")
    total = sum(len(v) for v in by_course.values())
    print(f"Loaded {total} LOs from {len(by_course)} courses -> {out}")


if __name__ == "__main__":
    main()
