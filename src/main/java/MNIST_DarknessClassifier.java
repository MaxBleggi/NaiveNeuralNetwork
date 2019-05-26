/**
 * A naive implementation of a classifier for recognizing handwritten digits from MNIST data set.
 * The Program classifies digits based on their "darkness" with the idea being that 1 is less dark than 8/
 *
 * So it selects the a digit with the closest average darkness as the input
 *
 * It first trains the classifier, then it applies the classifier to MNIST test data
 *
 */
public class MNIST_DarknessClassifier {

    /**
     *
     */
    public MNIST_DarknessClassifier() {

    }

    /**
     * Return a default dictionary whose keys are digits 0-9
     * For each digit, compute a value which is the average darkness of training images containing that digit.
     * The darkness for any particular image is just the sum of the darknesses for each pixel
     */
    public void avgDarkness() {

    }

    /**
     * Returns the digit whose average darkness in the training data is the closest to the darkness of the image
     */
    public void guessDigit() {

    }
}
