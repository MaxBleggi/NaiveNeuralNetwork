package Model;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *  Let TrainingData be a 2-tuples X = (x,y) such that
 *      { x | [x] is 784-dimensional array corresponding to the 28x28 image Matrix }
 *      { y | [y] is 10-dimensional array corresponding to the unit vector for the correct image }
 */
public class TrainingData {
    private INDArray x;
    private INDArray y;

    public TrainingData(INDArray x, INDArray y) {
        this.x = x;
        this.y = y;
    }

    public INDArray getX() {
        return x;
    }

    public INDArray getY() {
        return y;
    }
}
