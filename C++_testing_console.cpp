#include <iostream>
#include <vector>
#include <algorithm> 
#include <random>

// TODO
// 1 - Make it so it can have different amount of nodes, per layer.
//      Right now, it can only have x nr of layers, with y nr of nodes in each. aka only rectangle
//      Could be fixed, by having 3 dimentions?
// 
// 2 - Chosen activation function.
//      A paramater for creating the network, should be to choose a activation function
//          function pointer??
// 
// 3 - 

struct Node
{
    std::vector<double> weights;
    double bias;
};

class NeuralNetwork
{
private:
   
    const int nrOfInputNodes = 1; // 1 input
    const int nrOfOutputNodes = 1;
    const int nrOfHiddenLayers = 2; // 2 layers
    const int nrOfNodes = 2; // 2 nodes per layer

    double (NeuralNetwork::* activationFunction)(double);
    double (NeuralNetwork::* activFuncDerivative)(double);


    std::vector<std::vector<Node>> hiddenNodes;
    std::vector<Node> outputNodes;



   
    double sigmoid(double value) {
        return 1.0 / (1.0 + exp(-value));
    }

    double sigmoidDerivative(double value) {
        return sigmoid(value) * (1.0 - sigmoid(value));
    }

    double relU(double value) {
        return std::max(0.0, value);
    }

    double relUDerivative(double value) {
        return value > 0 ? 1.0 : 0.0;
    }

    double tanh_function(double value) {
        return (exp(value) - exp(-value)) / (exp(value) + exp(-value));
    }

    double tanh_derivative(double value) {
        double tanh_x = tanh_function(value);
        return 1.0 - tanh_x * tanh_x;
    }

    void initilizeWeightsAndBiases() {
        std::random_device random;
        std::mt19937 generator(random());
        std::uniform_real_distribution<> distribution(-1.0, 1.0);

        hiddenNodes.resize(nrOfHiddenLayers);
        for (int i = 0; i < nrOfHiddenLayers; ++i) {
            hiddenNodes[i].resize(nrOfNodes);
        }

        for (size_t i = 0; i < nrOfHiddenLayers; i++)
        {
            for (size_t j = 0; j < nrOfNodes; j++)
            {
                if (i == 0)
                {
                    hiddenNodes[i][j].weights.resize(nrOfInputNodes);
                    for (size_t k = 0; k < nrOfInputNodes; k++)
                    {
                        hiddenNodes[i][j].weights[k] = distribution(generator);
                    }
                }
                else
                {
                    hiddenNodes[i][j].weights.resize(nrOfNodes);
                    for (size_t k = 0; k < nrOfNodes; k++)
                    {
                        hiddenNodes[i][j].weights[k] = distribution(generator);
                    }
                }
                hiddenNodes[i][j].bias = distribution(generator);
            }
        }

        outputNodes.resize(nrOfOutputNodes);

        for (size_t i = 0; i < nrOfOutputNodes; i++)
        {
            outputNodes[i].weights.resize(nrOfNodes);
            outputNodes[i].bias = distribution(generator);
            for (size_t j = 0; j < outputNodes[i].weights.size(); j++)
            {
                outputNodes[i].weights[j] = distribution(generator);
            }
        }

    }

    std::vector<std::vector<double>> feedForward(std::vector<double> inputs) {
        std::vector<std::vector<double>> allOutputValues;  

        for (double input : inputs) {
            std::vector<double> currentInput{ input };

            for (size_t i = 0; i < nrOfHiddenLayers; i++)
            {
                std::vector<double> outputLayer(nrOfNodes, 0.0);  // current layer/input layer for the next layer
                for (size_t j = 0; j < nrOfNodes; j++)
                {
                    double sum = 0;
                    for (size_t k = 0; k < hiddenNodes[i][j].weights.size(); k++)
                    {
                        sum += (currentInput[k] * hiddenNodes[i][j].weights[k]);
                    }
                    sum += hiddenNodes[i][j].bias;
                    outputLayer[j] = (this->*activationFunction)(sum);                    
                }
                currentInput = outputLayer;
            }

            std::vector<double> outputValues(nrOfOutputNodes, 0.0);
            for (size_t i = 0; i < nrOfOutputNodes; i++)
            {
                double sum = 0;
                for (size_t j = 0; j < currentInput.size(); j++)
                {
                    sum += (currentInput[j] * outputNodes[i].weights[j]);
                }
                sum += outputNodes[i].bias;
                outputValues[i] = (this->*activationFunction)(sum);
            }

            allOutputValues.push_back(outputValues);  
        }

        return allOutputValues;
    }


    void backPropagate(double input){
        // TODO
    }

public:

    NeuralNetwork()
    {
        initilizeWeightsAndBiases();
        activationFunction = &NeuralNetwork::relU; // works?? maybe idk 
        activFuncDerivative = &NeuralNetwork::relUDerivative;
        std::cout << "got here 1" << std::endl;
    }


    void train() {
        // TODO
    }

    double predict(double input) {
        return -1.0; // TODO
    }
};

int main() {
   
    NeuralNetwork nn;
  
    return 0;
}