package projetACS;

import java.io.File;
// New Import: Class for loading saved models
import org.deeplearning4j.util.ModelSerializer; 
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class Predict {
    
    // Define a constant for your saved model file path
    private static final String MODEL_PATH = 
        "C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/TrainedModels/vgg16-transfer1.zip"; 
        
    public static void main(String[] args) throws Exception {
        
        // 1. Load the trained ComputationGraph model
        ComputationGraph model = ModelSerializer.restoreComputationGraph(new File(MODEL_PATH), true);
        System.out.println("Model loaded successfully from: " + MODEL_PATH);
        
        // 2. Load the image and run prediction
        INDArray img = loadImage(new File("C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/croc.jpg"), 224, 224);
        
        // Use the loaded model for prediction
        INDArray out = model.outputSingle(img); 

        System.out.println(out);
        System.out.println("Predicted: " + Nd4j.argMax(out, 1).getInt(0));
    }

    // This method needs to be inside the Predict class or declared as static outside of main
    public static INDArray loadImage(File file, int h, int w) throws Exception {
        NativeImageLoader loader = new NativeImageLoader(h, w, 3);
        INDArray img = loader.asMatrix(file);

        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(img);

        return img;
    }
}