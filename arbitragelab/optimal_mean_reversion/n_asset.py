# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name, too-many-instance-attributes
import numpy as np

class N_assets:
    """
    """

    def __init__(self):
        self.data = None
        self.X = None  # Portfolio price
        self.T = None  # Number of observations
        self.B = None  # Projection matrix
        self.gamma = None  # Penalizing constraint
        self.max_assets = None

    def allocate(self, data, w_0, delta_t, iterations=50, max_assets=weights.shape[0], gamma=0):
        """
        """

        self.data = data

        self.T = int(self.x.shape[0] - 1)

        self.B = np.eye(self.T) - np.ones((self.T, self.T)) / self.T

        self.delta_t = delta_t

        b0, b1 = self._b()
        if self._gamma_constraint(b0=b0, b1=b1):
            self.gamma = gamma
        else:
            raise Exception("Gamma is too large")

        self.max_assets = max_assets

        solution = self._solver(w=w0, a=np.random.rand(), iterations=iterations)

        c = solution[0][0]

        a = solution[0][1]

        f = solution[0][2]

        weights = solution[1][0]

        theta = sum((data[1:, :] - c * data[:-1, :]) @ weights) / (self.T * (1 - c))

        mu = (1 - c) / delta_t

        sigma_square = a / delta_t

        return theta, mu, sigma, weights

    def _norm(x):
        """
        """

        output = np.linalg.norm(x)
        return output

    def _b(self, x):
        """
        """

        output = [self.B @ x[:-1],
                  self.B @ x[1:]]

        return output

    def _gamma_constraint(self, b0, b1):
        """
        """

        output = True

        condition = lambda gamma: (self._norm(b0) ** 4
                                   - 4 * gamma * (self._norm(b0) ** 2 * self._norm(b1) ** 2) - (b1.T @ b0) ** 2)

        if condition(self.gamma) < 0:
            output = False

        return output

    def _simplex(w, h=1):
        """
        """

        n = w.shape[0]

        output = np.zeros((n, 1))

        summ_term = 0

        box = True

        for i in range(n):
            box = box and (w[i] >= 0 and w[i] <= 1)
            summ_term += w[i]

        if box and summ_term == h:
            output = w

        sorted_indices = np.argsort(w.T)[0]

        for i in range(n):

            alpha = (summ_term - h) / (n - i + 2)

            if alpha <= w[sorted_indices[i]]:

                for j in range(i, n):
                    output[sorted_indices[i]] = w[sorted_indices[i]] - alpha
                    break
            else:

                sum -= w[sorted_indices[i]]

        return output

    @static
    def _linesearch(x, gradient, f, prox, s, tau, tolerance=1e-7):
        """
        """
        output = x

        if isinstance(output, np.array):

            while np.greater_equal(f(output), f(x)).all() and s >= tolerance:
                output = prox(x - s * gradient)
                s *= tau
        else:
            while f(output) >= f(x) and s >= tolerance:
                output = prox(x - s * gradient)
                s *= tau

        return output

    def _c(self, b0, b1, a):
        """
        """
        output = (b0.T @ b1 - self.T * a * self.gamma) / self._norm(b0) ** 2

        return output

    def _A(self, c):
        """
        """
        output = self.data[1:, :] - c * self.data[:-1, :]

        return output

    def _loss(self, w, a):
        """
        """
        x = self.data @ w

        b0, b1 = self._b(x)

        c = self._c(b0=b0, b1=b1, a=a)

        A = self.A(c)

        return (np.log(a) / 2 + self._norm(self.B @ A @ w) ** 2 / (2 * self.T * a)
                + self.gamma * c - self.max_assets / 2 * self._norm(w) ** 2)

    def _solver(self, w, a, iterations):
        """
        """
        results = np.zeros((iter, 3))
        weights = np.zeros((iter, w.shape[0]))

        for i in range(iterations):

            x = self.data @ w

            b0, b1 = self._b(x)

            c = self._c(b0=b0, b1=b1, a=a)

            if b0.T @ b1 - self.T * a * self.gamma < 0:
                raise Exception("Gamma value is too large.")

            A = self._A(c)

            if self.gamma == 0:
                # If gamma is equal to zero the goal function doesn't depend on a

                # Define function that calculates optimal a with respect to w
                a = lambda w: self._norm(self.B @ A @ w) ** 2 / self.T

                # Define gradient value with respect to w
                dw = (((self.B @ A).T @ (self.B @ A) @ w) / self._norm(self.B @ A @ w) ** 2 - self.max_assets * w)

                # Define function f with respect to w
                f_w = lambda w: self._loss(w, a(w))

                # Establish a prox function for the w variable
                prox = lambda z: np.sign(z) * self._simplex(abs(z))

                w = self._linesearch(w, dw, f_w, prox, 1, 0.1)

                results[i, :] = [c, a(w), f_w(w)]

                weights[i, :] = w

            else:
                # Since gamma is not equal to zero the goal function still depends on a

                # Defining the gradient with respect to a
                da = 1 / (2 * a) - self._norm(B @ (x[1:] - c * x[:-1])) ** 2 / (2 * self.T * a ** 2)

                # Defining the f function with respect to a
                f_a = lambda a: self._loss(w, a)

                # Establish a prox function for the a variable
                prox = lambda z: max(z, 1e-10)

                a = self._linesearch(a, da, f_a, prox, 1, 0.1)

                # Define gradient value with respect to w
                dw = (((self.B @ A).T @ (self.B @ A) @ w) / (self.T * a) - self.max_assets * w)

                # Define the f function with respect to w
                f_w = lambda w: self._loss(w, a)

                # Establish a prox function for the w variable
                prox = lambda z: np.sign(z) * self._simplex(abs(z))

                w = self._linesearch(w, dw, f_w, 1, 0.1)

                results[i, :] = [c, a(w), f_w(w)]

                weights[i, :] = w

            return results[-1, :], weights[-1, :]
