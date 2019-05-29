package Model;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Let ValidationData and TestData both be lists containing 10,000 2-tuples X' = (x,y') such that
 *      { x | [x] is 784-dimensional array corresponding to the 28x28 image Matrix }
 *      { y' | 0 >= y' > 10 | representing the digit value of x }
 */
public class ValidationData {
    private DoubleMatrix1D x;
    private int y;

    public ValidationData(DoubleMatrix1D x, int y) {
        this.x = x;
        this.y = y;
    }

    public DoubleMatrix1D getX() {
        return x;
    }

    public int getY() {
        return y;
    }
}
