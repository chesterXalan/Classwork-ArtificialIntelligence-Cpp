#include <iostream>
#include <cmath>
using namespace std;


void Sigmoid(double x, double& y) {
    y = 1 / (1 + exp(-x));
}

void BackpropMmt(double(*W1)[3], double* W2, double(*X)[3], double* D) {
    double alpha = 0.9;
    double beta = 0.9;
    size_t N = 4;

    double mmt1[4][3], mmt2[4];
    double x[3], d, v1, y1[4], v, y;
    double e, e1, delta, delta1[4], dW1, dW2;

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++)
            mmt1[i][j] = 0;
        mmt2[i] = 0;
    }

    for (size_t k = 0; k < N; k++) {
        for (size_t i = 0; i < 3; i++)
            x[i] = X[k][i];
        d = D[k];

        for (size_t i = 0; i < 4; i++) {
            v1 = 0;
            for (size_t j = 0; j < 3; j++)
                v1 += W1[i][j] * x[j];
            Sigmoid(v1, y1[i]);
        }

        v = 0;
        for (size_t i = 0; i < 4; i++)
            v += W2[i] * y1[i];
        Sigmoid(v, y);

        e = d - y;
        delta = y * (1 - y) * e;
        for (size_t i = 0; i < 4; i++) {
            e1 = W2[i] * delta;
            delta1[i] = y1[i] * (1 - y1[i]) * e1;
        }

        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 3; j++) {
                dW1 = alpha * delta1[i] * x[j];
                mmt1[i][j] = dW1 + beta * mmt1[i][j];
                W1[i][j] += mmt1[i][j];
            }
            dW2 = alpha * delta * y1[i];
            mmt2[i] = dW2 + beta * mmt2[i];
            W2[i] += mmt2[i];
        }
    }
}

int main()
{
    srand(0);
    double X[4][3] = { {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1} };
    double D[4] = { 0, 1, 1, 0 };
    double W1[4][3], W2[4];

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++)
            W1[i][j] = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
        W2[i] = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
    }

    for (size_t i = 0; i < 10000; i++) //train
        BackpropMmt(W1, W2, X, D);

    size_t N = 4;
    double x[3], v1, y1[4], v, y;

    for (size_t k = 0; k < N; k++) { // inference
        for (size_t i = 0; i < 3; i++)
            x[i] = X[k][i];

        for (size_t i = 0; i < 4; i++) {
            v1 = 0;
            for (size_t j = 0; j < 3; j++)
                v1 += W1[i][j] * x[j];
            Sigmoid(v1, y1[i]);
        }

        v = 0;
        for (size_t i = 0; i < 4; i++)
            v += W2[i] * y1[i];
        Sigmoid(v, y);

        cout << "y" << k << " = " << y << endl;
    }
}
