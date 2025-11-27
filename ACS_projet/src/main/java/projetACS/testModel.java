package projetACS;

import java.io.File;


import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;


public class testModel {
    // -------------------------------------------------------------------------
    // You MUST set these values according to your dataset
    // -------------------------------------------------------------------------
    public static int height = 224;
    public static int width = 224;
    public static int channels = 3;
    public static int batchSize = 16;
    public static long seed = 1234;

    private final static String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    public static void main(String[] args) throws Exception {

        Random rng = new Random(seed);

        // -------------------------------------------------------------------------
        // 1. Load dataset directory
        // -------------------------------------------------------------------------
        File parentDir = new File("C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/Dataset");
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // -------------------------------------------------------------------------
        // 2. Train/Test split
        // -------------------------------------------------------------------------
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        InputSplit[] split = fileSplit.sample(pathFilter, 0, 1.0);
        InputSplit trainData = split[0];
        InputSplit testData = split[1];

        // -------------------------------------------------------------------------
        // 3. Prepare test data loader
        // -------------------------------------------------------------------------
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testData);

        int numLabels = testRR.getLabels().size();

        DataSetIterator testIter = new RecordReaderDataSetIterator(
                testRR, batchSize, 1, numLabels
        );

        // -------------------------------------------------------------------------
        // 4. Load your trained model
        // -------------------------------------------------------------------------
        String modelPath = "C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/TrainedModels/vgg16-transfer3.zip";
        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelPath);
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);

        // -------------------------------------------------------------------------
        // 5. Evaluate model on the complete test set
        // -------------------------------------------------------------------------
        Evaluation eval = new Evaluation(numLabels);

        while (testIter.hasNext()) {
            var ds = testIter.next();
            var output = model.outputSingle(ds.getFeatures());
            eval.eval(ds.getLabels(), output);
        }

        System.out.println(eval.stats());
        
//        Evaluation eval = model.evaluate(testIter);
//
//        System.out.println("-------------- EVALUATION RESULTS ----------------");
//        System.out.println(eval.stats());

        // Print labels
        System.out.println("Labels: " + testRR.getLabels());
    }
}
