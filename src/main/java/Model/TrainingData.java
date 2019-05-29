package Model;

import cern.colt.matrix.DoubleMatrix1D;

/**
 *  Let TrainingData be a 2-tuples X = (x,y) such that
 *      { x | [x] is 784-dimensional array corresponding to the 28x28 image Matrix }
 *      { y | [y] is 10-dimensional array corresponding to the unit vector for the correct image }
 */
public class TrainingData {
    private DoubleMatrix1D x;
    private DoubleMatrix1D y;

    public TrainingData(DoubleMatrix1D x, DoubleMatrix1D y) {
        this.x = x;
        this.y = y;
    }

    public DoubleMatrix1D getX() {
        return x;
    }

    public DoubleMatrix1D getY() {
        return y;
    }
}
