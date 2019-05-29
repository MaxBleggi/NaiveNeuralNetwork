package Model;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Let DataTuple be a 2-tuple DT such that:
 *      DT is a 2-tuple (x, y) where
 *      x is an array such that each entry is an n-dimensional array (Matrix W) with 784 values.
 *          This represents the the 784 pixels in each image. Therefore, x can be abstractly represented
 *          by a 28x28 matrix W. This will become more important in the NeuralNetwork where we can manipulate
 *          this convenient representation using linear algebra in order to simplify operations.
 *
 *      y is an n-dimensional array (Vector V) where each entry is a single digit range(0-9)
 *
 */
public class DataTuple {

    private DoubleMatrix1D[] x;
    private DoubleMatrix1D y;
    private int labels;
    private int nRows;
    private int nCols;

    public DataTuple(DoubleMatrix1D[] x, DoubleMatrix1D y, int labels) {
        this.x = x;
        this.y = y;
        this.labels = labels;
    }

    public DoubleMatrix1D[] getX() {
        return x;
    }

    public DoubleMatrix1D getY() {
        return y;
    }

    public int getLabels() {
        return labels;
    }
}
