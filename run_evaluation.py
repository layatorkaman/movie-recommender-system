from pathlib import Path

from src.data.loader import load_ratings
from src.evaluation.evaluation import evaluate_model_comparison


def main():
    project_root = Path(__file__).resolve().parent

    ratings_path = (
        project_root
        / "data"
        / "raw"
        / "ml-latest-small"
        / "ratings.csv"
    )

    ratings = load_ratings(ratings_path)

    results = evaluate_model_comparison(
        ratings,
        test_size=0.2,
        random_state=42,
        min_common_ratings=3,
        max_test_samples=500,
    )

    print("\nFair Model Comparison")
    print("=" * 60)

    for model_name in ["Baseline", "User-Based", "Item-Based"]:
        print(f"\n{model_name}:")
        for metric, value in results[model_name].items():
            print(f"{metric}: {value:.4f}")

    print("\nPrediction Coverage:")
    for model_name, count in results["coverage"].items():
        percentage = results["coverage_percentage"][model_name]
        print(
            f"{model_name}: {count} / {results['total_test_samples']} "
            f"({percentage:.2f}%)"
        )
    print("\nExecution Time:")
    for model_name, seconds in results["execution_time"].items():
        print(f"{model_name}: {seconds:.4f} seconds")

    print(f"\nCommon evaluated samples: {results['evaluated_samples']}")
    print(f"Skipped samples: {results['skipped_samples']}")


if __name__ == "__main__":
    main()