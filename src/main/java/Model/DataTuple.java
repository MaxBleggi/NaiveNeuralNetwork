package Model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

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

    private INDArray[] x;
    private INDArray y;
    private int label;
    private int nRows;
    private int nCols;

    public DataTuple(INDArray[] x, INDArray y) {
        this.x = x;
        this.y = y;
    }

    public INDArray[] getX() {
        return x;
    }

    public INDArray getY() {
        return y;
    }
}
