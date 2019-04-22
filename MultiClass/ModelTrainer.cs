using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MultiClass
{
    class ModelTrainer
    {
        MLContext _mlContext = null;
        string _filePath;
        IDataView _testDataView;
        
        public ModelTrainer(MLContext mLContext, string basePath)
        {
            _mlContext = mLContext;
            _filePath = basePath;
        }

        public void BuildAndTrain()
        {
            _testDataView = _mlContext.Data.LoadFromTextFile<NewsItem>($"{_filePath}/Data/test_data.tsv", hasHeader: true);

            int k = 5;
            bool fileThere = false;
            do
            {
                string trainFilePath = $"{_filePath}/Data/train_data{k}.tsv";
                fileThere = File.Exists(trainFilePath);
                if (fileThere)
                {
                    Console.WriteLine("training - " + k);
                    var trainingDataView = _mlContext.Data.LoadFromTextFile<NewsItem>(trainFilePath, hasHeader: true);

                    Console.WriteLine("  SDCA");
                    buildAndTrainSDCA(trainingDataView, GetPipeline(), k);
                    Console.WriteLine("  LR");
                    buildAndTrainLR(trainingDataView, GetPipeline(), k);

                    k += 5;
                }
            }
            while (fileThere);
        }

        void evaluateModel(IDataView testDataView, ITransformer model, string description, string report)
        {
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));
            List<string> reportVals = new List<string>();

            string v = $"*************************************************************************************************************";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*       Metrics for Multi-class Classification model - {description} - Test Data     ";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*------------------------------------------------------------------------------------------------------------";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*       LogLoss:          {testMetrics.LogLoss:#.###}";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}";
            reportVals.Add(v);
            Console.WriteLine(v);

            v = $"*************************************************************************************************************";
            reportVals.Add(v);
            Console.WriteLine(v);

            File.WriteAllLines(report, reportVals);
        }

        void buildAndTrainFT(IDataView trainingDataView, IEstimator<ITransformer> pipeline, int k)
        {
            string results = $"{_filePath}/Results/eval_FT{k}.txt";
            string path = $"{_filePath}/Models/model_FT{k}.zip";
            if (!File.Exists(results) || !File.Exists(path))
            {
                var trainingPipeline = pipeline
                    .Append(_mlContext
                    .Regression.Trainers.FastTree(DefaultColumnNames.Label, DefaultColumnNames.Features))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var trainedModel = trainingPipeline.Fit(trainingDataView);
                saveModel(trainedModel, path);
                evaluateModel(_testDataView, trainedModel, $"FT{k}", results);
            }            
        }

        void buildAndTrainLR(IDataView trainingDataView, IEstimator<ITransformer> pipeline, int k)
        {
            string results = $"{_filePath}/Results/eval_LR{k}.txt";
            string path = $"{_filePath}/Models/model_LR{k}.zip";
            if (!File.Exists(results) || !File.Exists(path))
            {
                var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers
                    .LogisticRegression(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var trainedModel = trainingPipeline.Fit(trainingDataView);
                saveModel(trainedModel, path);
                evaluateModel(_testDataView, trainedModel, $"FT{k}", results);
            }
        }

        void buildAndTrainSDCA(IDataView trainingDataView, IEstimator<ITransformer> pipeline, int k)
        {
            string results = $"{_filePath}/Results/eval_SD{k}.txt";
            string path = $"{_filePath}/Models/model_SD{k}.zip";
            if (!File.Exists(results) || !File.Exists(path))
            {
                var trainingPipeline = pipeline
                    .Append(_mlContext.MulticlassClassification.Trainers
                        .StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var trainedModel = trainingPipeline.Fit(trainingDataView);
                saveModel(trainedModel, path);
                evaluateModel(_testDataView, trainedModel, $"FT{k}", results);
            }
        }

        private void saveModel(TransformerChain<KeyToValueMappingTransformer> trainedModel, string path)
        {            
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
                _mlContext.Model.Save(trainedModel, fs);
        }

        public IEstimator<ITransformer> GetPipeline()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Feed", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }
    }
}
