using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MultiClass.Processors
{    
    class ModelLBME : ModelProcessor
    {
        public ModelLBME(Configuration config)
        {
            modelConfig = config;
            description = "Lbfgs MaximumEntropy";
            abbreviation = "LB";
        }
        protected override ITransformer TrainModel(int k, IDataView trainingDataView)
        {
            string results = $"{modelConfig.path}/Results/eval_{abbreviation}{k}.txt";
            string path = $"{modelConfig.path}/Models/model_{abbreviation}{k}.zip";
            var trainingPipeline = modelConfig.pipeline
                .Append(modelConfig.mlContext.MulticlassClassification.Trainers
                .LbfgsMaximumEntropy("Label", "Features"))
                .Append(modelConfig.mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.Write($"  training");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            return trainedModel;
        }
    }
}