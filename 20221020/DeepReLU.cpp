#include <iostream>
#include <cmath>
using namespace std;


void ReLU(double x, double& y) {
    if (x > 0)
        y = x;
    else
        y = 0;
}

void Softmax(double* x, double* y, size_t n) {
    double ex_sum = 0;

    for (size_t i = 0; i < n; i++)
        ex_sum += exp(x[i]);
    for (size_t i = 0; i < n; i++)
        y[i] = exp(x[i]) / ex_sum;
}

void DeepReLU(double(*W1)[25], double(*W2)[20], double(*W3)[20], double(*W4)[20], double(*X)[5][5], double(*D)[5]) {
    double alpha = 0.01;
    size_t N = 5;

    double x[25], d[5], v1[20], v2[20], v3[20], y1[20], y2[20], y3[20], v[5], y[5];
    double e[5], e3[20], e2[20], e1[20], delta[5], delta3[20], delta2[20], delta1[20], dW4, dW3, dW2, dW1;
    double W4_trans[20][5], W3_trans[20][20], W2_trans[20][20];

    for (size_t k = 0; k < N; k++) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++)
                x[i * 5 + j] = X[k][i][j];
            d[i] = D[k][i];
        }

        for (size_t i = 0; i < 20; i++) {
            v1[i] = 0;
            for (size_t j = 0; j < 25; j++)
                v1[i] += W1[i][j] * x[j];
            ReLU(v1[i], y1[i]);
        }
        for (size_t i = 0; i < 20; i++) {
            v2[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v2[i] += W2[i][j] * y1[j];
            ReLU(v2[i], y2[i]);
        }
        for (size_t i = 0; i < 20; i++) {
            v3[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v3[i] += W3[i][j] * y2[j];
            ReLU(v3[i], y3[i]);
        }
        for (size_t i = 0; i < 5; i++) {
            v[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v[i] += W4[i][j] * y3[j];
        }
        Softmax(v, y, 5);

        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 20; j++)
                W4_trans[j][i] = W4[i][j]; // W4 transpose
        }
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                W3_trans[j][i] = W3[i][j]; // W3 transpose
                W2_trans[j][i] = W2[i][j]; // W2 transpose
            }
        }

        for (size_t i = 0; i < 5; i++) {
            e[i] = d[i] - y[i];
            delta[i] = e[i];
        }
        for (size_t i = 0; i < 20; i++) {
            e3[i] = 0;
            for (size_t j = 0; j < 5; j++)
                e3[i] += W4_trans[i][j] * delta[j];
            if (v3[i] > 0)
                delta3[i] = e3[i];
            else
                delta3[i] = 0;
        }
        for (size_t i = 0; i < 20; i++) {
            e2[i] = 0;
            for (size_t j = 0; j < 20; j++)
                e2[i] += W3_trans[i][j] * delta3[j];
            if (v2[i] > 0)
                delta2[i] = e2[i];
            else
                delta2[i] = 0;
        }
        for (size_t i = 0; i < 20; i++) {
            e1[i] = 0;
            for (size_t j = 0; j < 20; j++)
                e1[i] += W2_trans[i][j] * delta2[j];
            if (v1[i] > 0)
                delta1[i] = e1[i];
            else
                delta1[i] = 0;
        }

        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 20; j++) {
                dW4 = alpha * delta[i] * y3[j];
                W4[i][j] += dW4;
            }
        }
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                dW3 = alpha * delta3[i] * y2[j];
                W3[i][j] += dW3;
            }
        }
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                dW2 = alpha * delta2[i] * y1[j];
                W2[i][j] += dW2;
            }
        }
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 25; j++) {
                dW1 = alpha * delta1[i] * x[j];
                W1[i][j] += dW1;
            }
        }
    }
}

void random(double& W) {
    W = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
}

int main() {
    srand(0);
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
    double W1[20][25], W2[20][20], W3[20][20], W4[5][20];

    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 25; j++)
            random(W1[i][j]);
        for (size_t j = 0; j < 20; j++) {
            random(W2[i][j]);
            random(W3[i][j]);
        }
    }
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 20; j++)
            random(W4[i][j]);
    }

    for (size_t i = 0; i < 10000; i++) //train
        DeepReLU(W1, W2, W3, W4, X, D);

    size_t N = 5;
    double x[25], v1[20], v2[20], v3[20], y1[20], y2[20], y3[20], v[5], y[5];

    for (size_t k = 0; k < N; k++) { // inference
        for (size_t i = 0; i < 5; i++)
            for (size_t j = 0; j < 5; j++)
                x[i * 5 + j] = X[k][i][j];

        for (size_t i = 0; i < 20; i++) {
            v1[i] = 0;
            for (size_t j = 0; j < 25; j++)
                v1[i] += W1[i][j] * x[j];
            ReLU(v1[i], y1[i]);
        }
        for (size_t i = 0; i < 20; i++) {
            v2[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v2[i] += W2[i][j] * y1[j];
            ReLU(v2[i], y2[i]);
        }
        for (size_t i = 0; i < 20; i++) {
            v3[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v3[i] += W3[i][j] * y2[j];
            ReLU(v3[i], y3[i]);
        }
        for (size_t i = 0; i < 5; i++) {
            v[i] = 0;
            for (size_t j = 0; j < 20; j++)
                v[i] += W4[i][j] * y3[j];
        }
        Softmax(v, y, 5);

        cout << "y" << k << " = " << endl;
        for (size_t i = 0; i < 5; i++)
            cout << y[i] << endl;
        cout << endl;
    }
}
