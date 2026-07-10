"""Small public demo for printing the conceptual HD-Diff pipeline."""

from hd_diff import HDDiffConcept


def main() -> None:
    model = HDDiffConcept()
    print("HD-Diff conceptual pipeline:")
    for index, step in enumerate(model.pipeline(), start=1):
        print(f"{index}. {step}")


if __name__ == "__main__":
    main()
