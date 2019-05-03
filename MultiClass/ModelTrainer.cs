using Microsoft.ML;
using MultiClass.Processors;
using System;
using System.Collections.Generic;
using System.IO;

namespace MultiClass
{
    class ModelTrainer
    {
        MLContext _mlContext = null;
        string _filePath;

        List<ModelProcessor> processors = new List<ModelProcessor>();

        public ModelTrainer(MLContext mLContext, string basePath)
        {
            _mlContext = mLContext;
            _filePath = basePath;
            
            var config = new Configuration()
            {
                mlContext = _mlContext,
                path = _filePath,
                pipeline = GetPipeline(),                
                testDataView = _mlContext.Data.LoadFromTextFile<NewsItem>($"{_filePath}/Data/test_data.tsv", hasHeader: true)
            };
            
            processors.Add(new ModelSCDAME(config));
            processors.Add(new ModelSCDANC(config));
            processors.Add(new ModelLBME(config));
        }

        public void BuildAndTrain()
        {
            int k = 5;
            bool fileThere = false;
            
            do
            {
                string trainFilePath = $"{_filePath}/Data/train_data{k}.tsv";
                fileThere = File.Exists(trainFilePath);
                if (fileThere)
                {
                    var trainingDataView = _mlContext.Data.LoadFromTextFile<NewsItem>(trainFilePath, hasHeader: true);

                    foreach (var p in processors)
                    {                        
                        p.ProcessModel(k, trainingDataView);
                    }
                    k += 5;
                }
            }
            while (fileThere);
        }

        protected IEstimator<ITransformer> GetPipeline()
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
