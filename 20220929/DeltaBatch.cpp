#include <iostream>
#include <cmath>
using namespace std;


double Sigmoid(double x) {
    double y = 1 / (1 + exp(-x));

    return y;
}


void DeltaBatch(double W[3], double X[4][3], double D[4]) {
    double alpha = 0.9;
    double dWsum[3], dWavg[3];
    int N = 4;

    for (int i = 0; i < 3; i++) {
        dWsum[i] = 0;
    }

    for (int k = 0; k < N; k++) {
        double x[3];
        double d, v, y, e, delta, dW;

        for (int i = 0; i < 3; i++) {
            x[i] = X[k][i];
        }
        d = D[k];

        v = 0;
        for (int i = 0; i < 3; i++) {
            v += W[i] * x[i];
        }

        y = Sigmoid(v);
        e = d - y;

        delta = y * (1 - y) * e;
        for (int i = 0; i < 3; i++) {
            dW = alpha * delta * x[i]; // delta rule
            dWsum[i] += dW;
        }
    }
    for (int i = 0; i < 3; i++) {
        dWavg[i] = dWsum[i] / N;
        W[i] += dWavg[i];
    }
}


int main()
{
    srand(0);
    double X[4][3] = { {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1} };
    double D[4] = { 0, 0, 1, 1 };
    double W[3];

    for (int i = 0; i < 3; i++) {
        W[i] = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1;
    }

    for (int i = 0; i < 40000; i++) { //train
        DeltaBatch(W, X, D);
    }

    int N = 4; // inference
    for (int k = 0; k < N; k++) {
        double x[3];
        double v, y;

        for (int i = 0; i < 3; i++) {
            x[i] = X[k][i];
        }

        v = 0;
        for (int i = 0; i < 3; i++) {
            v += W[i] * x[i];
        }

        y = Sigmoid(v);
        cout << "y" << k << " = " << y << endl;
    }
}
