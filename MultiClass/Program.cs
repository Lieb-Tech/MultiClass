using Microsoft.Azure.Documents.Client;
using Microsoft.Azure.Documents.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MultiClass
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "news_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "news_test.tsv");
        
        private static MLContext _mlContext;    
        private static PredictionEngine<NewsItem, SectionPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        /*
         *https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification
         */

        private static CosmosDB client = new CosmosDB();

        static void Main(string[] args)
        {
            client.OpenConnection();

            _mlContext = new MLContext(seed: 0);

            if (!File.Exists(_trainDataPath))
            {
                CreateTextFiles().Wait();
            }
            _trainingDataView = _mlContext.Data.LoadFromTextFile<NewsItem>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();
            // var trainingPipeline1 = BuildAndTrainSDCAModel(_trainingDataView, pipeline);
            // var trainingPipeline2 = BuildAndTrainBNModel(_trainingDataView, pipeline);
            var trainingPipeline3 = BuildAndTrainLRModel(_trainingDataView, pipeline);
            Evaluate();

            SaveModelAsFile(_mlContext, _trainedModel, "LR");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model, string prefix)
        {
            string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", prefix + "_feed_model.zip");

            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        public static void Evaluate()
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<NewsItem>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static IEstimator<ITransformer> BuildAndTrainSDCAModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {            
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers
                .StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainedModel.CreatePredictionEngine<NewsItem, SectionPrediction>(_mlContext);

            var prediction = _predEngine.Predict(new NewsItem()
            {
                Title = "Elizabeth Warren builds largest U.S. presidential campaign staff",
                Description = "U.S. Senator Elizabeth Warren has hired the largest campaign staff in the run-up to the 2020 presidential election, quickly building a payroll that far exceeds her Democratic rivals, according to disclosures filed with the Federal Elections Commission",
            });

            return trainingPipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainBNModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers
                .NaiveBayes(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainedModel.CreatePredictionEngine<NewsItem, SectionPrediction>(_mlContext);

            var prediction = _predEngine.Predict(new NewsItem()
            {
                Title = "Elizabeth Warren builds largest U.S. presidential campaign staff",
                Description = "U.S. Senator Elizabeth Warren has hired the largest campaign staff in the run-up to the 2020 presidential election, quickly building a payroll that far exceeds her Democratic rivals, according to disclosures filed with the Federal Elections Commission",
            });

            return trainingPipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainLRModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers
                .LogisticRegression(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainedModel.CreatePredictionEngine<NewsItem, SectionPrediction>(_mlContext);

            var prediction = _predEngine.Predict(new NewsItem()
            {
                Title = "Elizabeth Warren builds largest U.S. presidential campaign staff",
                Description = "U.S. Senator Elizabeth Warren has hired the largest campaign staff in the run-up to the 2020 presidential election, quickly building a payroll that far exceeds her Democratic rivals, according to disclosures filed with the Federal Elections Commission",
            });

            return trainingPipeline;
        }

        private async static Task CreateTextFiles()
        {
            if (!File.Exists(_trainDataPath))
            {
                FeedOptions queryOptions = new FeedOptions { MaxItemCount = 500, EnableCrossPartitionQuery = true };
                var uri = UriFactory.CreateDocumentCollectionUri("liebfeeds", "newsfeed");

                var qry = "SELECT c.id, c.title, c.description, c.partionKey, c.siteSection FROM c WHERE c.siteSection <> null";
                var docQry = client.GetDocumentQuery("newsfeed", qry, queryOptions).AsDocumentQuery();

                var vals = new List<CosmosNewsItem>();

                while (docQry.HasMoreResults)
                {
                    var z = await docQry.ExecuteNextAsync<CosmosNewsItem>();
                    vals.AddRange(z.Select(v => v).ToList());
                }

                File.WriteAllText(_trainDataPath, "ID\tFeed\tSiteSection\tTitle\tDescription\r\n");
                File.AppendAllLines(_trainDataPath, vals.OrderByDescending(v => v.id).Take(15000).Select(v => $"{v.id}\t{v.partionKey}\t{v.siteSection}\t{v.title}\t{v.description}").ToList());

                File.WriteAllText(_testDataPath, "ID\tFeed\tSiteSection\tTitle\tDescription\r\n");
                File.AppendAllLines(_testDataPath, vals.OrderByDescending(v => v.id).Take(100).Select(v => $"{v.id}\t{v.partionKey}\t{v.siteSection}\t{v.title}\t{v.description}").ToList());
                string s = "";
            }
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            // _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "SiteSection", outputColumnName: "Label")
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Feed", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }
    }
}
