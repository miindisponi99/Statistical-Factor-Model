import matplotlib.pyplot as plt
import numpy as np

class ComponentsAnalysis:
    def __init__(self, returns):
        self.returns = returns
        self.R = returns.values
        self.t, self.n = returns.shape
        self.Q_tilde = self.calculate_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.perform_eigendecomposition()
        self.explained_variance_ratio = self.calculate_explained_variance_ratio()
        self.cumulative_explained_variance = (
            self.calculate_cumulative_explained_variance()
        )

    def calculate_covariance_matrix(self):
        return (
            (1 / self.t) * self.R @ self.R.T
        )  # asset returns covariance matrix (in t x t not n x n, nello screenshot assume R che ha t come colonne)

    def perform_eigendecomposition(self):
        return np.linalg.eigh(self.Q_tilde)

    def calculate_explained_variance_ratio(self):
        return self.eigenvalues[::-1] / np.sum(self.eigenvalues)

    def calculate_cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance_ratio)

    def plot_scree(self):
        plt.figure(figsize=(20, 6))
        plt.plot(
            np.arange(1, len(self.eigenvalues) + 1),
            self.eigenvalues[::-1],
            "o-",
            markersize=8,
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Eigenvalue")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.show()

    def plot_cumulative_explained_variance(self):
        plt.figure(figsize=(20, 6))
        plt.plot(
            np.arange(1, len(self.eigenvalues) + 1),
            self.cumulative_explained_variance,
            "o-",
            markersize=8,
        )
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.axhline(y=0.90, color="r", linestyle="--")
        plt.grid(True)
        plt.show()


class APCA(ComponentsAnalysis):
    def __init__(self, returns, convergence_threshold=1e-3, max_iterations=1000):
        super().__init__(returns)
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.m = self.number_factors(0.90)
        self.U_m_final, self.F_final, self.B_final, self.Gamma_final = self.iterative_estimation()

    def number_factors(self, threshold):
        return np.searchsorted(self.cumulative_explained_variance, threshold) + 1

    def iterative_estimation(self):
        previous_Delta_squared = np.inf

        for iteration in range(self.max_iterations):
            U_m = self.eigenvectors[:, -self.m :]
            F = U_m.T  # Factor returns
            B = self.R.T @ U_m  # Factor exposures
            Gamma = self.R.T - B @ F  # Specific returns
            Delta_squared = (1 / self.t) * np.diag(
                Gamma @ Gamma.T
            )  # Specific covariance matrix

            if np.all(
                np.abs(Delta_squared - previous_Delta_squared)
                < self.convergence_threshold
            ):
                # print(f"Converged after {iteration + 1} iterations")
                break

            previous_Delta_squared = Delta_squared
            Delta_inv = np.diag(1 / np.sqrt(Delta_squared))
            R_star = Delta_inv @ self.R.T
            Q_tilde_star = (1 / self.n) * R_star.T @ R_star
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(Q_tilde_star)

        else:
            print("Did not converge within the maximum number of iterations")

        U_m_final = self.eigenvectors[:, -self.m :]
        F_final = U_m_final.T
        B_final = self.R.T @ U_m_final
        Gamma_final = self.R.T - B_final @ F_final

        return U_m_final, F_final, B_final, Gamma_final