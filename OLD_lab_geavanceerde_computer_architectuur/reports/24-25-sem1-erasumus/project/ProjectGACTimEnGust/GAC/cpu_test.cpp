#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>

#define MAX_INPUT_SIZE 140  // Maximum size of the input vector
#define MAX_OUTPUT_SIZE 140 // Maximum size of the output vector

#define MAX(a, b) ((a) > (b) ? (a) : (b))

//void transformMatrix(double source[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE],
//                     double dest[MAX_OUTPUT_SIZE][MAX_INPUT_SIZE]) {
//    for (int i = 0; i < MAX_INPUT_SIZE; ++i) {
//        for (int j = 0; j < MAX_OUTPUT_SIZE; ++j) {
//            dest[j][MAX_INPUT_SIZE - 1 - i] = source[i][j];
//        }
//    }
//}

double calculateMSE(const double input1[MAX_INPUT_SIZE], const double input2[MAX_INPUT_SIZE]) {
    double mse = 0.0;
    for (int i = 0; i < MAX_INPUT_SIZE; i++) {
        mse += (input1[i] - input2[i]) * (input1[i] - input2[i]);
    }

    // Return the mean squared error
    mse /= MAX_INPUT_SIZE;
    return mse;
}

// Function to load a matrix (weights) from a CSV file
void load_matrix(const char *filename, double matrix[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (file == nullptr) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%lf", &matrix[i][j]);
            if (j < cols - 1) {
                fscanf(file, ";"); // To skip the comma between values in CSV
            }
        }
    }
    fclose(file);
}

// Function to load a vector (bias) from a CSV file
void load_vector(const char *filename, double vector[MAX_INPUT_SIZE], int size) {
    FILE *file = fopen(filename, "r");
    if (file == nullptr) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    // Read the values line by line
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &vector[i]) != 1) {
            printf("Error reading value at index %d\n", i);
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}

// Function to calculate the matrix-vector multiplication and add bias
void forward_pass(const double input[MAX_INPUT_SIZE], double weights[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE],
                  const double bias[MAX_OUTPUT_SIZE], double output[MAX_OUTPUT_SIZE], int input_size, int output_size) {

    // Initialize output to biases
    for (int j = 0; j < output_size; ++j) {
        output[j] = bias[j];

        for (int i = 0; i < input_size; ++i) { // Iterate over rows of A^T (input dimensions)
            output[j] += input[i] * weights[j][i];
        }
    }
}

void relu(double output[MAX_OUTPUT_SIZE], int size){
    for (int i = 0; i < size; ++i) {
        output[i] = MAX(0, output[i]);
    }
}

void sigmoid(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        // Sigmoid formula: 1 / (1 + e^(-x))
        vector[i] = 1.0 / (1.0 + exp(-vector[i]));
    }
}

void print_vector(double vector[MAX_OUTPUT_SIZE], int size, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < size; i++) {
        std::cout << vector[i] << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Define sizes for input and output vectors for each layer
    int encoder_input_size = 140;
    int encoder_output_size_1 = 140;
    int encoder_output_size_2 = 128;
    int encoder_output_size_3 = 64;
    int encoder_output_size_4 = 32;

    int decoder_input_size = 64;
    int decoder_output_size_1 = 128;
    int decoder_output_size_2 = 140;
    int decoder_output_size_3 = 140;
    int decoder_output_size_4 = 140;

    // Load the input vector (e.g., ecg.csv)
    double vector[MAX_INPUT_SIZE];
    const char *filename = "input_signal.csv";
    load_vector(filename, vector, MAX_INPUT_SIZE);

    // Declare weight and bias matrices for encoder and decoder layers
    double encoder_weights_1[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], encoder_weights_2[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], encoder_weights_3[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], encoder_weights_4[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE];
    double encoder_bias_1[MAX_OUTPUT_SIZE], encoder_bias_2[MAX_OUTPUT_SIZE], encoder_bias_3[MAX_OUTPUT_SIZE], encoder_bias_4[MAX_OUTPUT_SIZE];

    double decoder_weights_1[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], decoder_weights_2[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], decoder_weights_3[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE], decoder_weights_4[MAX_INPUT_SIZE][MAX_OUTPUT_SIZE];
    double decoder_bias_1[MAX_OUTPUT_SIZE], decoder_bias_2[MAX_OUTPUT_SIZE], decoder_bias_3[MAX_OUTPUT_SIZE], decoder_bias_4[MAX_OUTPUT_SIZE];

    // Load encoder layer weights and biases
    load_matrix("encoder_layer_1_weights.csv", encoder_weights_1, 140, 140);
    load_vector("encoder_layer_1_biases.csv", encoder_bias_1, 140);
    load_matrix("encoder_layer_3_weights.csv", encoder_weights_2, 128, 140);
    load_vector("encoder_layer_3_biases.csv", encoder_bias_2, 128);
    load_matrix("encoder_layer_5_weights.csv", encoder_weights_3, 64, 128);
    load_vector("encoder_layer_5_biases.csv", encoder_bias_3, 64);
    load_matrix("encoder_layer_7_weights.csv", encoder_weights_4, 32, 64);
    load_vector("encoder_layer_7_biases.csv", encoder_bias_4, 32);

    // Load decoder layer weights and biases
    load_matrix("decoder_layer_1_weights.csv", decoder_weights_1, 64, 32);
    load_vector("decoder_layer_1_biases.csv", decoder_bias_1, 64);
    load_matrix("decoder_layer_3_weights.csv", decoder_weights_2, 128, 64);
    load_vector("decoder_layer_3_biases.csv", decoder_bias_2, 128);
    load_matrix("decoder_layer_5_weights.csv", decoder_weights_3, 140, 128);
    load_vector("decoder_layer_5_biases.csv", decoder_bias_3, 140);
    load_matrix("decoder_layer_7_weights.csv", decoder_weights_4, 140, 140);
    load_vector("decoder_layer_7_biases.csv", decoder_bias_4, 140);

    // Output vectors after forward pass
    double encoder_output_1[MAX_OUTPUT_SIZE], encoder_output_2[MAX_OUTPUT_SIZE], encoder_output_3[MAX_OUTPUT_SIZE], encoder_output_4[MAX_OUTPUT_SIZE];
    double decoder_output_1[MAX_OUTPUT_SIZE], decoder_output_2[MAX_OUTPUT_SIZE], decoder_output_3[MAX_OUTPUT_SIZE], decoder_output_4[MAX_OUTPUT_SIZE];

    auto start = std::chrono::high_resolution_clock::now();

    // Perform forward pass through the encoder layers
    forward_pass(vector, encoder_weights_1, encoder_bias_1, encoder_output_1, encoder_input_size, encoder_output_size_1);
    relu(encoder_output_1, encoder_output_size_1);
//    print_vector(encoder_output_1, encoder_output_size_1, "Encoder Layer 1 Output");

    forward_pass(encoder_output_1, encoder_weights_2, encoder_bias_2, encoder_output_2, encoder_output_size_1, encoder_output_size_2);
    relu(encoder_output_2, encoder_output_size_2);
//    print_vector(encoder_output_2, encoder_output_size_2, "Encoder Layer 2 Output");

    forward_pass(encoder_output_2, encoder_weights_3, encoder_bias_3, encoder_output_3, encoder_output_size_2, encoder_output_size_3);
    relu(encoder_output_3, encoder_output_size_3);
//    print_vector(encoder_output_3, encoder_output_size_3, "Encoder Layer 3 Output");

    forward_pass(encoder_output_3, encoder_weights_4, encoder_bias_4, encoder_output_4, encoder_output_size_3, encoder_output_size_4);
//    print_vector(encoder_output_4, encoder_output_size_4, "Encoder Layer 4 Output");

    // Perform forward pass through the decoder layers
    forward_pass(encoder_output_4, decoder_weights_1, decoder_bias_1, decoder_output_1, decoder_input_size, decoder_output_size_1);
    relu(decoder_output_1, decoder_output_size_1);
//    print_vector(decoder_output_1, decoder_output_size_1, "Decoder Layer 1 Output");

    forward_pass(decoder_output_1, decoder_weights_2, decoder_bias_2, decoder_output_2, decoder_output_size_1, decoder_output_size_2);
    relu(decoder_output_2, decoder_output_size_2);
//    print_vector(decoder_output_2, decoder_output_size_2, "Decoder Layer 2 Output");

    forward_pass(decoder_output_2, decoder_weights_3, decoder_bias_3, decoder_output_3, decoder_output_size_2, decoder_output_size_3);
    relu(decoder_output_3, decoder_output_size_3);
//    print_vector(decoder_output_3, decoder_output_size_3, "Decoder Layer 3 Output");

    forward_pass(decoder_output_3, decoder_weights_4, decoder_bias_4, decoder_output_4, decoder_output_size_3, decoder_output_size_4);
    sigmoid(decoder_output_4, decoder_output_size_4);
//    print_vector(decoder_output_4, decoder_output_size_4, "Decoder Layer 4 Output");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Calculate MSE between the final output and the expected output
    double mse = calculateMSE(vector, decoder_output_4);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    return 0;
}
