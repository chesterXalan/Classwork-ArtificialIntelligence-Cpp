#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

void reverseInt(int& i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    i = ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readMnistImage(string filename, double**** images, size_t* size) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_cols, sizeof(n_cols));
        file.read((char*)&n_rows, sizeof(n_rows));
        reverseInt(magic_number);
        reverseInt(number_of_images);
        reverseInt(n_rows);
        reverseInt(n_cols);

        if (magic_number == 2051) {
            cout << "Read MNIST image data..." << endl;

            double*** array = new double** [number_of_images];
            for (size_t i = 0; i < number_of_images; i++) {
                array[i] = new double* [n_rows];
                for (size_t j = 0; j < n_rows; j++) {
                    array[i][j] = new double[n_cols];
                    for (size_t k = 0; k < n_cols; k++) {
                        unsigned char pixel = 0;
                        file.read((char*)&pixel, sizeof(pixel));
                        array[i][j][k] = (double)pixel/255;
                    }
                }
            }
            *images = array;
            size[0] = number_of_images;
            size[1] = n_rows;
            size[2] = n_cols;
        }
    }
}

void readMnistLabel(string filename, size_t** labels, size_t* size) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        reverseInt(magic_number);
        reverseInt(number_of_labels);

        if (magic_number == 2049) {
            cout << "Read MNIST label data..." << endl;

            size_t* array = new size_t[number_of_labels];
            for (size_t i = 0; i < number_of_labels; i++) {
                unsigned char value = 0;
                file.read((char*)&value, sizeof(value));
                array[i] = (size_t)value;
            }
            *labels = array;
            size[0] = number_of_labels;
        }
    }
}

void create3dArray(double**** a, size_t* a_size, double init = 0) {
    double*** array = new double** [a_size[0]];
    for (size_t i = 0; i < a_size[0]; i++) {
        array[i] = new double* [a_size[1]];
        for (size_t j = 0; j < a_size[1]; j++) {
            array[i][j] = new double[a_size[2]];
            for (size_t k = 0; k < a_size[2]; k++)
                array[i][j][k] = init;
        }
    }
    *a = array;
}

void create2dArray(double*** a, size_t* a_size, double init = 0) {
    double** array = new double* [a_size[0]];
    for (size_t i = 0; i < a_size[0]; i++) {
        array[i] = new double[a_size[1]];
        for (size_t j = 0; j < a_size[1]; j++)
            array[i][j] = init;
    }
    *a = array;
}

void create1dArray(double** a, size_t* a_size, double init = 0) {
    double* array = new double[a_size[0]];
    for (size_t i = 0; i < a_size[0]; i++)
        array[i] = init;
    *a = array;
}

void delete3dArray(double*** a, size_t* a_size) {
    for (size_t i = 0; i < a_size[0]; i++) {
        for (size_t j = 0; j < a_size[1]; j++)
            delete[] a[i][j];
        delete[] a[i];
    }
    delete[] a;
}

void delete2dArray(double** a, size_t* a_size) {
    for (size_t i = 0; i < a_size[0]; i++)
        delete[] a[i];
    delete[] a;
}

double ReLU(double x) {
    if (x > 0)
        return x;
    else
        return 0;
}

void Softmax(double* x, double* y, size_t n) {
    double ex_sum = 0;

    for (size_t i = 0; i < n; i++)
        ex_sum += exp(x[i]);
    for (size_t i = 0; i < n; i++)
        y[i] = exp(x[i]) / ex_sum;
}

void Conv2d(double** x, double** y, double** kernel, size_t* y_size, size_t* kernel_size, size_t stride = 1) {
    double conv;
    size_t a, b;

    for (size_t i = 0; i < y_size[1]; i++) {
        for (size_t j = 0; j < y_size[2]; j++) {
            conv = 0;
            a = i * stride;
            for (size_t r = 0; r < kernel_size[1]; r++) {
                b = j * stride;
                for (size_t c = 0; c < kernel_size[2]; c++) {
                    conv += x[a][b] * kernel[r][c];
                    b++;
                }
                a++;
            }
            y[i][j] = conv;
        }
    }
}

void Pool(double*** x, double*** y, size_t* x_size, size_t* y_size) {
    size_t filter_size[] = { x_size[0], 2, 2 }; // 20x2x2
    double*** filter = NULL;
    create3dArray(&filter, filter_size, 1.0 / (filter_size[1] * filter_size[2]));

    for (size_t i = 0; i < filter_size[0]; i++)
        Conv2d(x[i], y[i], filter[i], y_size, filter_size, 2);
}

void MnistConv(double*** W1, double** W5, double** Wo, double*** X, size_t* D, size_t* W1_size, size_t* W5_size, size_t* Wo_size, size_t* X_size, size_t* D_size) {
    double alpha = 0.01, beta = 0.95, temp;

    size_t N = D_size[0];
    size_t bsize = 100;

    // create arrays
    size_t blist_size = N / bsize; // 20
    size_t* blist = new size_t[blist_size];
    for (size_t i = 0; i < blist_size; i++)
        blist[i] = i * bsize;     

    size_t momentum1_size[] = { W1_size[0], W1_size[1], W1_size[2] }; // 20x9x9
    double*** momentum1 = NULL;
    create3dArray(&momentum1, momentum1_size);

    size_t momentum5_size[] = { W5_size[0], W5_size[1] }; // 100x2000
    double** momentum5 = NULL;
    create2dArray(&momentum5, momentum5_size);

    size_t momentumo_size[] = { Wo_size[0], Wo_size[1] }; // 10x100
    double** momentumo = NULL;
    create2dArray(&momentumo, momentumo_size);

    size_t dW1_size[] = { W1_size[0], W1_size[1], W1_size[2] }; // 20x9x9
    double*** dW1 = NULL;
    create3dArray(&dW1, dW1_size);

    size_t dW5_size[] = { W5_size[0], W5_size[1] }; // 100x2000
    double** dW5 = NULL;
    create2dArray(&dW5, dW5_size);

    size_t dWo_size[] = { Wo_size[0], Wo_size[1] }; // 10x100
    double** dWo = NULL;
    create2dArray(&dWo, dWo_size);

    size_t x_size[] = { X_size[1], X_size[2] }; // 28x28
    double** x = NULL;
    create2dArray(&x, x_size);

    size_t y1_size[] = { W1_size[0], x_size[0] - (W1_size[1] - 1), x_size[1] - (W1_size[2] - 1) }; // 20x20x20
    double*** y1 = NULL;
    create3dArray(&y1, y1_size);

    size_t y2_size[] = { y1_size[0], y1_size[1], y1_size[2] }; // 20x20x20
    double*** y2 = NULL;
    create3dArray(&y2, y2_size);

    size_t y3_size[] = { y2_size[0], y2_size[1] / 2, y2_size[2] / 2 }; // 20x10x10
    double*** y3 = NULL;
    create3dArray(&y3, y3_size);

    size_t y4_size[] = { y3_size[0] * y3_size[1] * y3_size[2] }; // 2000
    double* y4 = NULL;
    create1dArray(&y4, y4_size);

    size_t v5_size[] = { W5_size[0] }; // 100
    double* v5 = NULL;
    create1dArray(&v5, v5_size);

    size_t y5_size[] = { v5_size[0] }; // 100
    double* y5 = NULL;
    create1dArray(&y5, y5_size);
    
    size_t v_size[] = { Wo_size[0] }; // 10
    double* v = NULL;
    create1dArray(&v, v_size);

    size_t y_size[] = { v_size[0] }; // 10
    double* y = NULL;
    create1dArray(&y, y_size);

    size_t d_size[] = { 10 };
    double* d = NULL;
    create1dArray(&d, d_size);

    size_t e_size[] = { d_size[0] }; // 10
    double* e = NULL;
    create1dArray(&e, e_size);

    size_t delta_size[] = { e_size[0] }; // 10
    double* delta = NULL;
    create1dArray(&delta, delta_size);

    size_t e5_size[] = { Wo_size[1] }; // 100
    double* e5 = NULL;
    create1dArray(&e5, e5_size);

    size_t delta5_size[] = { e5_size[0] }; // 100
    double* delta5 = NULL;
    create1dArray(&delta5, delta5_size);

    size_t e4_size[] = { W5_size[1] }; // 2000
    double* e4 = NULL;
    create1dArray(&e4, e4_size);

    size_t e3_size[] = { y3_size[0], y3_size[1], y3_size[2] }; // 20x10x10
    double*** e3 = NULL;
    create3dArray(&e3, e3_size);

    size_t e2_size[] = { y2_size[0], y2_size[1], y2_size[2] }; // 20x20x20
    double*** e2 = NULL;
    create3dArray(&e2, e2_size);

    size_t W3_size[] = { y2_size[0], y2_size[1], y2_size[2] }; // 20x20x20
    double*** W3 = NULL;
    create3dArray(&W3, W3_size, 1.0 / (2 * 2));

    size_t delta2_size[] = { e2_size[0], e2_size[1], e2_size[2] }; // 20x20x20
    double*** delta2 = NULL;
    create3dArray(&delta2, delta2_size);

    size_t delta1_x_size[] = { W1_size[0], W1_size[1], W1_size[2] }; // 20x9x9
    double*** delta1_x = NULL;
    create3dArray(&delta1_x, delta1_x_size);
    

    for (size_t batch = 0; batch < blist_size; batch++) {
        for (size_t i = 0; i < dW1_size[0]; i++) {
            for (size_t j = 0; j < dW1_size[1]; j++) {
                for (size_t k = 0; k < dW1_size[2]; k++)
                    dW1[i][j][k] = 0;
            }
        }
        for (size_t i = 0; i < dW5_size[0]; i++) {
            for (size_t j = 0; j < dW5_size[1]; j++)
                dW5[i][j] = 0;
        }
        for (size_t i = 0; i < dWo_size[0]; i++) {
            for (size_t j = 0; j < dWo_size[1]; j++)
                dWo[i][j] = 0;
        }

        size_t begin = blist[batch];
        for (size_t data = begin; data < begin + bsize; data++) {
            for (size_t i = 0; i < x_size[0]; i++) {
                for (size_t j = 0; j < x_size[1]; j++)
                    x[i][j] = X[data][i][j];
            }

            for (size_t i = 0; i < y1_size[0]; i++)
                Conv2d(x, y1[i], W1[i], y1_size, W1_size);

            for (size_t i = 0; i < y2_size[0]; i++) {
                for (size_t j = 0; j < y2_size[1]; j++) {
                    for (size_t k = 0; k < y2_size[2]; k++)
                        y2[i][j][k] = ReLU(y1[i][j][k]);
                }
            }

            Pool(y2, y3, y2_size, y3_size);

            for (size_t i = 0; i < y3_size[0]; i++) {
                for (size_t j = 0; j < y3_size[1]; j++) {
                    for (size_t k = 0; k < y3_size[2]; k++)
                        y4[i * y3_size[1] * y3_size[2] + j * y3_size[2] + k] = y3[i][j][k];
                }
            }

            for (size_t i = 0; i < W5_size[0]; i++) {
                v5[i] = 0;
                for (size_t j = 0; j < W5_size[1]; j++)
                    v5[i] += W5[i][j] * y4[j];
                y5[i] = ReLU(v5[i]);
            }

            for (size_t i = 0; i < Wo_size[0]; i++) {
                v[i] = 0;
                for (size_t j = 0; j < Wo_size[1]; j++)
                    v[i] += Wo[i][j] * y5[j];
            }
            Softmax(v, y, v_size[0]);

            for (size_t i = 0; i < d_size[0]; i++)
                d[i] = 0;
            d[D[data]] = 1.0;

            for (size_t i = 0; i < delta_size[0]; i++) {
                e[i] = d[i] - y[i];
                delta[i] = e[i];
            }

            for (size_t i = 0; i < Wo_size[1]; i++) {
                e5[i] = 0;
                for (size_t j = 0; j < Wo_size[0]; j++)
                    e5[i] += Wo[j][i] * delta[j];
            }

            for (size_t i = 0; i < delta5_size[0]; i++) {
                if (y5[i] > 0) 
                    delta5[i] = e5[i];
                else
                    delta5[i] = 0;
            }

            for (size_t i = 0; i < W5_size[1]; i++) {
                e4[i] = 0;
                for (size_t j = 0; j < W5_size[0]; j++)
                    e4[i] += W5[j][i] * delta5[j];
            }

            size_t idx = 0;
            for (size_t i = 0; i < e3_size[0]; i++) {
                for (size_t j = 0; j < e3_size[1]; j++) {
                    for (size_t k = 0; k < e3_size[2]; k++)
                        e3[i][j][k] = e4[idx++];
                }
            }

            size_t kron_size[] = { 2, 2 };
            for (size_t i = 0; i < e3_size[0]; i++) {
                for (size_t j = 0; j < e3_size[1]; j++) {
                    for (size_t k = 0; k < e3_size[2]; k++) {
                        temp = e3[i][j][k] * W3[i][j][k];

                        for (size_t a = 0; a < kron_size[0]; a++) {
                            for (size_t b = 0; b < kron_size[1]; b++)
                                e2[i][j * kron_size[0] + a][k * kron_size[1] + b] = temp;
                        }
                    }
                }
            }

            for (size_t i = 0; i < delta2_size[0]; i++) {
                for (size_t j = 0; j < delta2_size[1]; j++) {
                    for (size_t k = 0; k < delta2_size[2]; k++) {
                        if (y2[i][j][k] > 0)
                            delta2[i][j][k] = e2[i][j][k];
                        else
                            delta2[i][j][k] = 0;
                    }
                }
            }

            for (size_t i = 0; i < delta1_x_size[0]; i++)
                Conv2d(x, delta1_x[i], delta2[i], delta1_x_size, delta2_size);
            
            for (size_t i = 0; i < dW1_size[0]; i++) {
                for (size_t j = 0; j < dW1_size[1]; j++) {
                    for (size_t k = 0; k < dW1_size[2]; k++)
                        dW1[i][j][k] += delta1_x[i][j][k];
                }
            }
            
            for (size_t i = 0; i < dW5_size[0]; i++) {
                for (size_t j = 0; j < dW5_size[1]; j++)
                    dW5[i][j] += delta5[i] * y4[j];
            }

            for (size_t i = 0; i < dWo_size[0]; i++) {
                for (size_t j = 0; j < dWo_size[1]; j++)
                    dWo[i][j] += delta[i] * y5[j];
            }
        }

        for (size_t i = 0; i < dW1_size[0]; i++) {
            for (size_t j = 0; j < dW1_size[1]; j++) {
                for (size_t k = 0; k < dW1_size[2]; k++)
                    dW1[i][j][k] /= bsize;
            }
        }
        for (size_t i = 0; i < dW5_size[0]; i++) {
            for (size_t j = 0; j < dW5_size[1]; j++)
                dW5[i][j] /= bsize;
        }
        for (size_t i = 0; i < dWo_size[0]; i++) {
            for (size_t j = 0; j < dWo_size[1]; j++)
                dWo[i][j] /= bsize;
        }

        for (size_t i = 0; i < momentum1_size[0]; i++) {
            for (size_t j = 0; j < momentum1_size[1]; j++) {
                for (size_t k = 0; k < momentum1_size[2]; k++) {
                    momentum1[i][j][k] = alpha * dW1[i][j][k] + beta * momentum1[i][j][k];
                    W1[i][j][k] += momentum1[i][j][k];
                }
            }
        }
        for (size_t i = 0; i < momentum5_size[0]; i++) {
            for (size_t j = 0; j < momentum5_size[1]; j++) {
                momentum5[i][j] = alpha * dW5[i][j] + beta * momentum5[i][j];
                W5[i][j] += momentum5[i][j];
            }
        }
        for (size_t i = 0; i < momentumo_size[0]; i++) {
            for (size_t j = 0; j < momentumo_size[1]; j++) {
                momentumo[i][j] = alpha * dWo[i][j] + beta * momentumo[i][j];
                Wo[i][j] += momentumo[i][j];
            }
        }
    }

    // delete arrays
    delete3dArray(momentum1, momentum1_size);
    delete2dArray(momentum5, momentum5_size);
    delete2dArray(momentumo, momentumo_size);
    delete3dArray(dW1, dW1_size);
    delete2dArray(dW5, dW5_size);
    delete2dArray(dWo, dWo_size);
    delete2dArray(x, x_size);
    delete3dArray(y1, y1_size);
    delete3dArray(y2, y2_size);
    delete3dArray(y3, y3_size);
    delete3dArray(e3, e3_size);
    delete3dArray(e2, e2_size);
    delete3dArray(W3, W3_size);
    delete3dArray(delta2, delta2_size);
    delete3dArray(delta1_x, delta1_x_size);
    delete[] blist, y4, v5, y5, v, y, d, e, delta, e5, delta5, e4;
}

double Random() {
    double W = 2 * ((double)rand() / (RAND_MAX + 1.0)) - 1.0;
    return W;
}

int main() {
    /*
    default_random_engine generator;
    uniform_real_distribution<double> rand_uni(0.0, 1.0);
    normal_distribution<double> rand_norm(0.0, 1.0);
    */
    size_t images_size[3] = { 0 }, labels_size[1] = { 0 };
    double*** images = NULL;
    size_t* labels = NULL;
    readMnistImage("t10k-images.idx3-ubyte", &images, images_size);
    readMnistLabel("t10k-labels.idx1-ubyte", &labels, labels_size);


    size_t train_num = 2000;
    size_t X1_size[] = { train_num ,images_size[1], images_size[2] }; // 2000x28x28
    double*** X1 = new double** [X1_size[0]];
    for (size_t i = 0; i < X1_size[0]; i++) {
        X1[i] = new double* [X1_size[1]];
        for (size_t j = 0; j < X1_size[1]; j++) {
            X1[i][j] = new double[X1_size[2]];
            for (size_t k = 0; k < X1_size[2]; k++)
                X1[i][j][k] = images[i][j][k];
        }
    }

    size_t D1_size[] = { train_num }; // 2000
    size_t* D1 = new size_t[D1_size[0]];
    for (size_t i = 0; i < D1_size[0]; i++)
        D1[i] = labels[i];

    size_t W1_size[] = { 20, 9, 9 };
    double*** W1 = new double** [W1_size[0]];
    for (size_t i = 0; i < W1_size[0]; i++) {
        W1[i] = new double* [W1_size[1]];
        for (size_t j = 0; j < W1_size[1]; j++) {
            W1[i][j] = new double[W1_size[2]];
            for (size_t k = 0; k < W1_size[2]; k++)
                W1[i][j][k] = Random() * 1e-1;
        }
    }

    size_t W5_size[] = { 100, 2000 };
    double** W5 = new double* [W5_size[0]];
    for (size_t i = 0; i < W5_size[0]; i++) {
        W5[i] = new double[W5_size[1]];
        for (size_t j = 0; j < W5_size[1]; j++)
            W5[i][j] = Random() * sqrt(10) / sqrt(W5_size[0] + W5_size[1]);
    }

    size_t Wo_size[] = { 10, 100 };
    double** Wo = new double* [Wo_size[0]];
    for (size_t i = 0; i < Wo_size[0]; i++) {
        Wo[i] = new double[Wo_size[1]];
        for (size_t j = 0; j < Wo_size[1]; j++)
            Wo[i][j] = Random() * sqrt(10) / sqrt(Wo_size[0] + Wo_size[1]);
    }

    // train
    cout << "\nStart training..." << endl;
    size_t epoch = 10;
    for (size_t i = 0; i < epoch; i++) {
        cout << "\r" << "Epoch: " << i+1 << "/" << epoch;
        MnistConv(W1, W5, Wo, X1, D1, W1_size, W5_size, Wo_size, X1_size, D1_size);
    }

    // test
    cout << "\n\nTraining complete. Start testing..." << endl;
    size_t test_num = 1000;
    size_t X2_size[] = { test_num ,images_size[1], images_size[2] }; // 1000x28x28
    double*** X2 = new double** [X2_size[0]];
    for (size_t i = 0; i < X2_size[0]; i++) {
        X2[i] = new double* [X2_size[1]];
        for (size_t j = 0; j < X2_size[1]; j++) {
            X2[i][j] = new double[X2_size[2]];
            for (size_t k = 0; k < X2_size[2]; k++)
                X2[i][j][k] = images[i + train_num][j][k];
        }
    }

    // create arrays
    size_t D2_size[] = { test_num }; // 1000
    size_t* D2 = new size_t[D2_size[0]];
    for (size_t i = 0; i < D2_size[0]; i++)
        D2[i] = labels[i + train_num];

    size_t x_size[] = { X2_size[1], X2_size[2] }; // 28x28
    double** x = NULL;
    create2dArray(&x, x_size);

    size_t y1_size[] = { W1_size[0], x_size[0] - (W1_size[1] - 1), x_size[1] - (W1_size[2] - 1) }; // 20x20x20
    double*** y1 = NULL;
    create3dArray(&y1, y1_size);

    size_t y2_size[] = { y1_size[0], y1_size[1], y1_size[2] }; // 20x20x20
    double*** y2 = NULL;
    create3dArray(&y2, y2_size);

    size_t y3_size[] = { y2_size[0], y2_size[1] / 2, y2_size[2] / 2,  }; // 20x10x10
    double*** y3 = NULL;
    create3dArray(&y3, y3_size);

    size_t y4_size[] = { y3_size[0] * y3_size[1] * y3_size[2] }; // 2000
    double* y4 = NULL;
    create1dArray(&y4, y4_size);

    size_t v5_size[] = { W5_size[0] }; // 100
    double* v5 = NULL;
    create1dArray(&v5, v5_size);

    size_t y5_size[] = { v5_size[0] }; // 100
    double* y5 = NULL;
    create1dArray(&y5, y5_size);

    size_t v_size[] = { Wo_size[0] }; // 10
    double* v = NULL;
    create1dArray(&v, v_size);

    size_t y_size[] = { v_size[0] }; // 10
    double* y = NULL;
    create1dArray(&y, y_size);


    double acc = 0;
    size_t pred;
    double max_p;
    for (size_t data = 0; data < test_num; data++) {
        for (size_t i = 0; i < x_size[0]; i++) {
            for (size_t j = 0; j < x_size[1]; j++)
                x[i][j] = X2[data][i][j];
        }

        for (size_t i = 0; i < y1_size[0]; i++)
            Conv2d(x, y1[i], W1[i], y1_size, W1_size );

        for (size_t i = 0; i < y2_size[0]; i++) {
            for (size_t j = 0; j < y2_size[1]; j++) {
                for (size_t k = 0; k < y2_size[2]; k++)
                    y2[i][j][k] = ReLU(y1[i][j][k]);
            }
        }

        Pool(y2, y3, y2_size, y3_size);

        for (size_t i = 0; i < y3_size[0]; i++) {
            for (size_t j = 0; j < y3_size[1]; j++) {
                for (size_t k = 0; k < y3_size[2]; k++)
                    y4[i * y3_size[1] * y3_size[2] + j * y3_size[2] + k] = y3[i][j][k];
            }
        }

        for (size_t i = 0; i < W5_size[0]; i++) {
            v5[i] = 0;
            for (size_t j = 0; j < W5_size[1]; j++)
                v5[i] += W5[i][j] * y4[j];
            y5[i] = ReLU(v5[i]);
        }

        for (size_t i = 0; i < Wo_size[0]; i++) {
            v[i] = 0;
            for (size_t j = 0; j < Wo_size[1]; j++)
                v[i] += Wo[i][j] * y5[j];
        }
        Softmax(v, y, v_size[0]);

        pred = 0;
        max_p = 0;
        for (size_t i = 0; i < y_size[0]; i++) {
            if (y[i] > max_p) {
                max_p = y[i];
                pred = i;
            }     
        }
        if (pred == D2[data])
            acc += 1.0;
    }
    acc /= test_num;
    cout << "Accuracy is " << acc * 100 << "%" << endl;

    // delete arrays
    delete3dArray(images, images_size);
    delete3dArray(X1, X1_size);
    delete3dArray(X2, X2_size);
    delete3dArray(W1, W1_size);
    delete2dArray(W5, W5_size);
    delete2dArray(Wo, Wo_size);
    delete2dArray(x, x_size);
    delete3dArray(y1, y1_size);
    delete3dArray(y2, y2_size);
    delete3dArray(y3, y3_size);
    delete[] labels, D1, D2, y4, v5, y5, v, y;
}
