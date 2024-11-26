package utils

func MSE(y, yPred float64) float64 {
    return 0.5 * (y - yPred) * (y - yPred)
}

func MSEDeriv(y, yPred float64) float64 {
    return yPred - y
}
