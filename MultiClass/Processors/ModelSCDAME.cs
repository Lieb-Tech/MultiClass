using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace MultiClass.Processors
{
    class ModelSCDAME : ModelProcessor
    {
        public ModelSCDAME(Configuration config)
        {
            modelConfig = config;
            description = "SDCA MaxEntrop";
            abbreviation = "SD";
    }
        protected override ITransformer TrainModel(int k, IDataView trainingDataView)
        {
            string results = $"{modelConfig.path}/Results/eval_{abbreviation}{k}.txt";
            string path = $"{modelConfig.path}/Models/model_{abbreviation}{k}.zip";
            var trainingPipeline = modelConfig.pipeline
                .Append(modelConfig.mlContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy("Label", "Features"))
                .Append(modelConfig.mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.Write($"  training");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            return trainedModel;
        }
    }
}
