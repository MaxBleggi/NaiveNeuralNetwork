import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipFile;

/**
 * Loads the MNIST image data set which is neccesary to train network
 */
public class MNIST_Loader {

    public MNIST_Loader() {

    }

    /**
     * Loads the MNIST TrainingData, TD such that
     *      TD is a 2-tuple (x, y) where
     *      x is a n-dimensional array with 50,000 entries. Where each entry is another n-dim. array with 784 values.
     *      This represents the the 784 pixels in each image.
     *      y us an n-dimensional array with with 50,000 entries. Where each entry is a single digit range(0-9)
     *
     */
    public void loadDataTrainingData(String file) {
        // read in from zip file

    }

    /**
     * Returns a tuple containing training data, validation data, and test data.
     * Alters the data to be more convenient to use for neural networks
     *
     * Training data is a list that contains 50,000 2-tuples (x,y). X is 784-dimensional numpy.ndarray
     * y is 10-dimensional numpy.ndarray
     *
     * Validation and Test data the same except 10,000
     */
    public void dataWrapper() {

    }

    /**
     * Returns a 10-dimensional unit vector with a 1 in the jth position and zeros elsewhere.
     * Used to convert a 0-9 digit into a corresponding desired output from the nnetwork
     */
    public void vectorizedResults() {

    }
}
