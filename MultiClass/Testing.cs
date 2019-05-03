using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MultiClass
{
    /*
    public class Testing
    {
        MLContext _mlContext;
        private static ITransformer _loadedModel;

        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineSD25 = null;
        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineLR25 = null;

        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineSD20 = null;
        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineLR20 = null;

        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineSD15 = null;
        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineLR15 = null;

        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineSD10 = null;
        private static PredictionEngine<MLNewsItem, SectionPrediction> _predEngineLR10 = null;

        string projectRootPath = ".";
        
        public Testing()
        {

        }

        public void InitModels()
        {
            _mlContext = new MLContext(seed: 0);

            Console.WriteLine("SD25");
            string _modelPath = projectRootPath + "/Models/SD25_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineSD25 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("SD20");
            _modelPath = projectRootPath + "/Models/SD20_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineSD20 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("SD15");
            _modelPath = projectRootPath + "/Models/SD15_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineSD15 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("SD10");
            _modelPath = projectRootPath + "/Models/SD10_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineSD10 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("LR25");
            _modelPath = projectRootPath + "/Models/LR25_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineLR25 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("LR20");
            _modelPath = projectRootPath + "/Models/LR20_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineLR20 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("LR15");
            _modelPath = projectRootPath + "/Models/LR15_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineLR15 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

            Console.WriteLine("LR10");
            _modelPath = projectRootPath + "/Models/LR10_feed_model.zip";
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                _loadedModel = _mlContext.Model.Load(stream);
            _predEngineLR10 = _loadedModel.CreatePredictionEngine<MLNewsItem, SectionPrediction>(_mlContext);

        }

        public void RunTests()
        {
            var cdb = new CosmosDB();
            Console.WriteLine("Getting data");
            var qry = cdb.GetDocumentQuery<CosmoseNewsItem>("newsfeed")
                 .OrderByDescending(z => z._ts)
                 // .Skip(25000)
                .Take(500)
                .ToList();

            var results = new List<TestResult>();
            Console.WriteLine("Processing");
            foreach (var n in qry)
            {
                MLNewsItem singleIssue = new MLNewsItem()
                {
                    Title = n.title,
                    Description = n.description
                };

                results.Add(new TestResult()
                {
                    sd25 = _predEngineSD25.Predict(singleIssue).Section,
                    lr25 = _predEngineLR25.Predict(singleIssue).Section,

                    sd20 = _predEngineSD20.Predict(singleIssue).Section,
                    lr20 = _predEngineLR20.Predict(singleIssue).Section,

                    sd15 = _predEngineSD15.Predict(singleIssue).Section,
                    lr15 = _predEngineLR15.Predict(singleIssue).Section,
                    
                    sd10 = _predEngineSD10.Predict(singleIssue).Section,
                    lr10 = _predEngineLR10.Predict(singleIssue).Section,

                    actual = n.partionKey
                });
            }

            var sd25 = results.Count(z => z.sd25 == z.actual);            
            var sd20 = results.Count(z => z.sd20 == z.actual);
            var sd15 = results.Count(z => z.sd15 == z.actual);
            var sd10 = results.Count(z => z.sd10 == z.actual);

            var lr10 = results.Count(z => z.lr10 == z.actual);
            var lr15 = results.Count(z => z.lr15 == z.actual);
            var lr20 = results.Count(z => z.lr20 == z.actual);
            var lr25 = results.Count(z => z.lr25 == z.actual);

            var aaz = "";
        }
    }
    */

    public class TestResult
    {
        public string sd10;
        public string sd15;
        public string sd20;
        public string sd25;

        public string lr10;
        public string lr15;
        public string lr20;
        public string lr25;

        public string actual;
    }

    public class CosmoseNewsItem
    {
        public string pubDate { get; set; }
        public string title { get; set; }
        public string partionKey { get; set; }
        public string description { get; set; }
        public string link { get; set; }
        public long _ts { get; set; }
    }

    public class MLNewsItem
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Feed { get; set; }
        [LoadColumn(2)]
        public string SiteSection { get; set; }
        [LoadColumn(3)]
        public string Title { get; set; }
        [LoadColumn(4)]
        public string Description { get; set; }
    }    
}