#include <iostream>
#include <cmath>
using namespace std;


double Sigmoid(double x) {
    double y = 1 / (1 + exp(-x));

    return y;
}


void DeltaSGD(double W[3], double X[4][3], double D[4]) {
    double alpha = 0.9;
    int N = 4;

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
            W[i] += dW;
        }
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

    for (int i = 0; i < 10000; i++) { //train
        DeltaSGD(W, X, D);       
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
