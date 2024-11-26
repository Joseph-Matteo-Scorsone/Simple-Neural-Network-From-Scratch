package main

import (
	types "NeuralNet/types"
	"fmt"
)

func main() {
	inputSize := 5
	hiddenSize := 4
	outputSize := 1
	learningRate := 0.01
	epochs := 10000

	timeSeriesData := [][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5},
		{0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5},
		{0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5},
		{0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5},
		{0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5},
	}
	targets := []float64{0.6, 0.7, 0.8, 0.9, 0.7}

	nn := types.NewNeuralNetwork(inputSize, hiddenSize, outputSize, learningRate)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range timeSeriesData {
			nn.Train(input, targets[i])
		}
		if epoch%1000 == 0 {
			predictions := nn.Predict(timeSeriesData[0])
			fmt.Printf("Epoch %d, Prediction: %f\n", epoch, predictions)
		}
	}

	// Final prediction
	prediction := nn.Predict(timeSeriesData[0])
	fmt.Printf("Final prediction: %f\n", prediction)
}