#include <iostream>
#include <cmath>
using namespace std;


void Sigmoid(double x, double& y) {
    y = 1 / (1 + exp(-x));
}

void Softmax(double *x, double *y, size_t n) {
    double ex_sum = 0;

    for (size_t i = 0; i < n; i++)
        ex_sum += exp(x[i]);
    for (size_t i = 0; i < n; i++)
        y[i] = exp(x[i]) / ex_sum;
}

void MultiClass(double(*W1)[25], double(*W2)[50], double(*X)[5][5], double(*D)[5]) {
    double alpha = 0.9;
    size_t N = 5;

    double x[25], d[5], v1[50], y1[50], v[5], y[5];
    double e[5], e1[50], delta[5], delta1[50], dW1, dW2;
    double W2_trans[50][5];

    for (size_t k = 0; k < N; k++) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++)
                x[i * 5 + j] = X[k][i][j];
            d[i] = D[k][i];
        }
        for (size_t i = 0; i < 50; i++) {
            v1[i] = 0;
            for (size_t j = 0; j < 25; j++)
                v1[i] += W1[i][j] * x[j];
            Sigmoid(v1[i], y1[i]);
        }
        for (size_t i = 0; i < 5; i++) {
            v[i] = 0;
            for (size_t j = 0; j < 50; j++)
                v[i] += W2[i][j] * y1[j];
        }
        Softmax(v, y, 5);

        for (size_t i = 0; i < 5; i++) {
            e[i] = d[i] - y[i];
            delta[i] = e[i];
        }
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 50; j++)
                W2_trans[j][i] = W2[i][j]; // W2 transpose
        }

        for (size_t i = 0; i < 50; i++) {
            e1[i] = 0;
            for (size_t j = 0; j < 5; j++)
                e1[i] += W2_trans[i][j] * delta[j];
            delta1[i] = y1[i] * (1 - y1[i]) * e1[i];
        }

        for (size_t i = 0; i < 50; i++) {
            for (size_t j = 0; j < 25; j++) {
                dW1 = alpha * delta1[i] * x[j];
                W1[i][j] += dW1;
            }
        }
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 50; j++) {
                dW2 = alpha * delta[i] * y1[j];
                W2[i][j] += dW2;
            }
        }
    }
}

int main() {
    srand(3);
    double X[5][5][5] = {
       {{0, 1, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}, // 1
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0}},
       {{1, 1, 1, 1, 0},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}, // 2
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}},
       {{1, 1, 1, 1, 0},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}, // 3
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 0}},
       {{0, 0, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 1, 0, 1, 0}, // 4
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 0}},
       {{1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 0}, // 5
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 0}}
    };
    double D[5][5] = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 1},
    };
    double W1[50][25], W2[5][50];

    for (size_t i = 0; i < 50; i++) {
        for (size_t j = 0; j < 25; j++)
            W1[i][j] = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
    }
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 50; j++)
            W2[i][j] = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
    }
    for (size_t i = 0; i < 10000; i++) //train
        MultiClass(W1, W2, X, D);

    size_t N = 5;
    double x[25], v1[50], y1[50], v[5], y[5];

    for (size_t k = 0; k < N; k++) { // inference
        for (size_t i = 0; i < 5; i++)
            for (size_t j = 0; j < 5; j++)
                x[i * 5 + j] = X[k][i][j];

        for (size_t i = 0; i < 50; i++) {
            v1[i] = 0;
            for (size_t j = 0; j < 25; j++)
                v1[i] += W1[i][j] * x[j];
            Sigmoid(v1[i], y1[i]);
        }
        for (size_t i = 0; i < 5; i++) {
            v[i] = 0;
            for (size_t j = 0; j < 50; j++)
                v[i] += W2[i][j] * y1[j];
        }
        Softmax(v, y, 5);

        cout << "y" << k << " = " << endl;
        for (size_t i = 0; i < 5; i++)
            cout << y[i] << endl;
        cout << endl;
    }
}
