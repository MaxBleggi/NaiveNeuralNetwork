import Model.DataTuple;
import Model.NNetworkInputLoad;
import Model.TrainingData;
import Model.ValidationData;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;

public class Driver {

    public static final String TEST_IMAGES = "src/main/resources/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS =  "src/main/resources/t10k-labels.idx1-ubyte";
    public static final String TRAINING_IMAGES =     "src/main/resources/train-images.idx3-ubyte";
    public static final String TRAINING_LABEL =     "src/main/resources/train-labels.idx1-ubyte";

    public static void main(String[] argv) {

        BasicConfigurator.configure();
        MNIST_Loader load = new MNIST_Loader();
        try {
            NNetworkInputLoad inputLoad = load.dataWrapper(TRAINING_IMAGES, TRAINING_LABEL, TEST_IMAGES, TEST_LABELS);
            TrainingData[] trainingData = inputLoad.getTrainingData();
            ValidationData[] testData = inputLoad.getTestData();
            System.out.println("Data from MNIST successfully loaded...");

            int[] sizes = {784, 28, 10};
            NNetwork nn = new NNetwork(sizes);
            nn.stochGradientDescent(trainingData, testData, 10,10, 0.5);


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
