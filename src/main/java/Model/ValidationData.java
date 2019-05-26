package Model;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Let ValidationData and TestData both be lists containing 10,000 2-tuples X' = (x,y') such that
 *      { x | [x] is 784-dimensional array corresponding to the 28x28 image Matrix }
 *      { y' | 0 >= y' > 10 | representing the digit value of x }
 */
public class ValidationData {
    private INDArray x;
    private int y;

    public ValidationData(INDArray x, int y) {
        this.x = x;
        this.y = y;
    }

    public INDArray getX() {
        return x;
    }

    public int getY() {
        return y;
    }
}
