from clean import load_and_clean_data
from model import train_and_evaluate
from plots import generate_plots


def main():
    # Step 1: Load and clean the dataset
    _, X, y = load_and_clean_data()

    # Step 2: Train models and print metrics
    rf_model, feature_names, y_test, y_preds = train_and_evaluate(X, y)

    # Step 3: Save evaluation plots to outputs/
    generate_plots(rf_model, feature_names, y_test, y_preds)


if __name__ == "__main__":
    main()
