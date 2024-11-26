package types

import (
	utils "NeuralNet/utils"
	"math/rand"
)

// Define the structure for the Neural Network
type NeuralNetwork struct {
	inputSize           int
	hiddenSize          int
	outputSize          int
	learningRate        float64
	weightsInputHidden  [][]float64
	weightsHiddenOutput []float64
	biasHidden          []float64
	biasOutput          float64
}

// Initialize the Neural Network with random weights and biases
func NewNeuralNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *NeuralNetwork {
	r := rand.New(rand.NewSource(1025))
	nn := &NeuralNetwork{
		inputSize:           inputSize,
		hiddenSize:          hiddenSize,
		outputSize:          outputSize,
		learningRate:        learningRate,
		weightsInputHidden:  make([][]float64, inputSize),
		weightsHiddenOutput: make([]float64, hiddenSize),
		biasHidden:          make([]float64, hiddenSize),
		biasOutput:          r.Float64(),
	}

	// Initialize weights and biases
	for i := 0; i < inputSize; i++ {
		nn.weightsInputHidden[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			nn.weightsInputHidden[i][j] = r.Float64()
		}
	}

	for j := 0; j < hiddenSize; j++ {
		nn.weightsHiddenOutput[j] = r.Float64()
		nn.biasHidden[j] = r.Float64()
	}

	return nn
}

// Feedforward step: Predict output for a given input
func (nn *NeuralNetwork) Predict(input []float64) float64 {
	// Hidden layer activations
	hiddenLayer := make([]float64, nn.hiddenSize)
	for j := 0; j < nn.hiddenSize; j++ {
		sum := nn.biasHidden[j]
		for i := 0; i < nn.inputSize; i++ {
			sum += input[i] * nn.weightsInputHidden[i][j]
		}
		hiddenLayer[j] = utils.ReLU(sum)
	}

	// Output layer (single neuron)
	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += hiddenLayer[j] * nn.weightsHiddenOutput[j]
	}

	return output
}

// Backpropagation and weight update step
func (nn *NeuralNetwork) Train(input []float64, target float64) {
	// Feedforward step
	hiddenLayer := make([]float64, nn.hiddenSize)
	for j := 0; j < nn.hiddenSize; j++ {
		sum := nn.biasHidden[j]
		for i := 0; i < nn.inputSize; i++ {
			sum += input[i] * nn.weightsInputHidden[i][j]
		}
		hiddenLayer[j] = utils.ReLU(sum)
	}

	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += hiddenLayer[j] * nn.weightsHiddenOutput[j]
	}

	// Compute the loss gradient (MSE)
	outputError := utils.MSEDeriv(target, output)

	// Backpropagation through the output layer
	for j := 0; j < nn.hiddenSize; j++ {
		nn.weightsHiddenOutput[j] -= nn.learningRate * outputError * hiddenLayer[j]
	}
	nn.biasOutput -= nn.learningRate * outputError

	// Backpropagation through the hidden layer
	hiddenErrors := make([]float64, nn.hiddenSize)
	for j := 0; j < nn.hiddenSize; j++ {
		hiddenErrors[j] = outputError * nn.weightsHiddenOutput[j] * utils.ReLUDeriv(hiddenLayer[j])
		for i := 0; i < nn.inputSize; i++ {
			nn.weightsInputHidden[i][j] -= nn.learningRate * hiddenErrors[j] * input[i]
		}
		nn.biasHidden[j] -= nn.learningRate * hiddenErrors[j]
	}
}
