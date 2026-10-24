import os
import numpy as np
import matplotlib.pyplot as plt


def generate_plots(model, feature_names, y_test, y_preds):
    # Create outputs/ folder automatically if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Actual vs. Predicted Alcohol Percentage
    plot1_path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.figure(figsize=(7, 5))
    plt.scatter(
        y_test, y_preds, alpha=0.6, color="royalblue", edgecolors="k", s=40
    )
    min_val = min(y_test.min(), y_preds.min())
    max_val = max(y_test.max(), y_preds.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal Fit (y = x)",
    )
    plt.title("Actual vs. Predicted Wine Alcohol Percentage (%)")
    plt.xlabel("Actual Alcohol (%)")
    plt.ylabel("Predicted Alcohol (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved {plot1_path}")

    # Plot 2: Feature Importance Bar Chart
    plot2_path = os.path.join(output_dir, "feature_importance.png")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    plt.barh(
        np.array(feature_names)[sorted_idx],
        importances[sorted_idx],
        color="teal",
        edgecolor="black",
    )
    plt.title(
        "Physicochemical Feature Importance for Predicting Wine Alcohol (%)"
    )
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Wine Physicochemical Properties")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved {plot2_path}")
