import numpy as np
import matplotlib.pyplot as plt


def simulate_classification_data_altered(decision_boundary=6, noise_level=0.1):
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Study hours (0-10)
    Y = (X > decision_boundary).astype(int)  # Pass if X > decision_boundary

    Y = np.where(np.random.rand(100, 1) < noise_level, 1 - Y, Y)

    # Plot
    plt.scatter(X, Y, label="Data Points", c=Y, cmap="bwr")
    plt.axvline(X=decision_boundary, color="gray", linestyle="--", label="Decision Boundary")
    plt.xlabel("Study Hours")
    plt.ylabel("Pass (1) / Fail (0)")
    plt.title(f"Classification Data (Boundary: X > {decision_boundary}, Noise: {noise_level})")
    plt.legend()
    plt.show()

    return X,Y


if __name__ == "__main__":
    simulate_classification_data_altered(decision_boundary=6, noise_level=0.1)
