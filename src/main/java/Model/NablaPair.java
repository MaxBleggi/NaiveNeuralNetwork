package Model;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public class NablaPair {
    DoubleMatrix1D[] nabla_b;
    DoubleMatrix2D[] nabla_w;

    public NablaPair(DoubleMatrix1D[] nabla_b, DoubleMatrix2D[] nabla_w) {
        this.nabla_b = nabla_b;
        this.nabla_w = nabla_w;
    }

    public DoubleMatrix1D[] getNabla_b() {
        return nabla_b;
    }

    public DoubleMatrix2D[] getNabla_w() {
        return nabla_w;
    }
}
