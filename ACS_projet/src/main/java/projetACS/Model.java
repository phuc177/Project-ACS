package projetACS;
import java.io.File;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.core.storage.StatsStorage;


import projetACS.DataPipeline;


public class Model {
	
	public static void main(String[] args) throws Exception {
		File rootDir = new File("C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/Dataset"); 
		final int height = 224;
		final int width = 224;
		final int channels = 3;
		final int batchSize = 16;
		final long seed = 1234;
		final double splitTrainTest = 80; // 80% for training, 20% for testing
		Random rng = new Random(seed);
		final int epochs = 8;
		
		DataPipeline dataset = new DataPipeline(height, width, channels, batchSize, seed);
        DataPipeline.TrainTestData data = dataset.load(
                rootDir,     // <-- YOUR DATASET PATH
                splitTrainTest                           // 80% train / 20% test
        );

        var trainIter = data.trainIter;
        var testIter  = data.testIter;
        int numClasses = 7;
		
		// create the network
		ZooModel model = VGG16.builder().build();
		@SuppressWarnings("deprecation")
		// load the pretrained weight
		ComputationGraph pretrainedNet = (ComputationGraph) model.initPretrained(PretrainedType.IMAGENET);
		
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .updater(new Nesterovs(5e-5))
	            .seed(seed)
	            .build();
		
		
		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
			    .fineTuneConfiguration(fineTuneConf)
			              .setFeatureExtractor("fc2")
			              .removeVertexKeepConnections("predictions") 
			              .addLayer("predictions", 
			        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			                        .nIn(4096).nOut(numClasses)
			                        .weightInit(WeightInit.XAVIER)
			                        .activation(Activation.SOFTMAX).build(), "fc2")
			              .build();

		// Setup the UI server
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network will store the training statistics
		// InMemoryStatsStorage is simplest, but file-based is also an option
		StatsStorage statsStorage = new InMemoryStatsStorage();

		// Attach the StatsStorage instance to the UI server
		uiServer.attach(statsStorage);

		// Add the StatsListener to your model
		vgg16Transfer.setListeners(new StatsListener(statsStorage));

		
        // -----------------------------
        // TRAINING LOOP
        // -----------------------------

        for (int epoch = 1; epoch <= epochs; epoch++) {

            System.out.println("/n===== EPOCH " + epoch + "/" + epochs + " =====");
            vgg16Transfer.fit(trainIter);
            trainIter.reset();

            // Evaluation
            Evaluation eval = vgg16Transfer.evaluate(testIter);
            System.out.println(eval.stats());
            testIter.reset();
        }

        // -----------------------------
        // SAVE MODEL
        // -----------------------------
        File saveFile = new File("C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/TrainedModels/vgg16-transfer3.zip");
        vgg16Transfer.save(saveFile, true);
        System.out.println("Model saved to: " + saveFile.getAbsolutePath());
		
	}
}
