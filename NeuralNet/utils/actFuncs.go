package utils

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDeriv(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDeriv(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
