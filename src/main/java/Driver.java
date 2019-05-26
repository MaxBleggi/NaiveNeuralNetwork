import Model.DataTuple;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;

public class Driver {

    private static final String TRAINING_IMAGES = "src/main/resources/t10k-images.idx3-ubyte";
    private static final String TRAINING_LABEL = "src/main/resources/t10k-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "main/resources/train-images.idx3-ubyte";
    private static final String TEST_LABELS = "main/resources/train-labels.idx1-ubyte";

    public static void main(String[] argv) {

        BasicConfigurator.configure();
        MNIST_Loader load = new MNIST_Loader();
        try {
            DataTuple dt = load.loadDataTrainingData(TRAINING_IMAGES, TRAINING_LABEL);
            System.out.println("#### IMG Data ####");
            System.out.println(Arrays.toString(dt.getX()));

            System.out.println("#### Label Data ####");
            System.out.println(dt.getY());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
