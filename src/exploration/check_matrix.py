
from src.data.loader import load_ratings
from src.recommenders.user_based import build_user_item_matrix


DATA_PATH = "../../data/raw/ml-latest-small/ratings.csv"


def main():
    ratings = load_ratings(DATA_PATH)
    matrix = build_user_item_matrix(ratings)

    total_cells = matrix.size
    rated_cells = matrix.notna().sum().sum()
    density = rated_cells / total_cells

    print("Matrix shape:", matrix.shape)
    print("Total cells:", total_cells)
    print("Rated cells:", rated_cells)
    print("Density:", round(density, 4))
    print("Sparsity:", round(1 - density, 4))


if __name__ == "__main__":
    main()
