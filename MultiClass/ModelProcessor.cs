using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MultiClass
{
    abstract class ModelProcessor
    {
        abstract protected ITransformer TrainModel(int k, IDataView trainingDataView);

        protected Configuration modelConfig;
        protected string description;
        protected string abbreviation;

        public void ProcessModel(int k, IDataView trainingDataView)
        {
            string path = modelConfig.path + $"/Models/model_{abbreviation}{k}.zip";
            if (File.Exists(path))
            {
                Console.WriteLine($"Already exists {abbreviation} - {k}k");
            }
            else
            {
                Console.Write($"Processing - {abbreviation} - {k}k");
                var model = TrainModel(k, trainingDataView);
                EvaluateModel(model, k);
                SaveModel(model, modelConfig.testDataView.Schema, k);
                Console.WriteLine(" completed");
            }
        }

        protected void SaveModel(ITransformer model, DataViewSchema schema, int k)
        {
            Console.Write($"  saving");
            string path = modelConfig.path + $"/Models/model_{abbreviation}{k}.zip";
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
                modelConfig.mlContext.Model.Save(model, schema, fs);
        }

        protected void EvaluateModel(ITransformer model, int k)
        {
            Console.Write($"  evaluating");
            var testMetrics = modelConfig.mlContext.MulticlassClassification.Evaluate(model.Transform(modelConfig.testDataView));
            List<string> reportVals = new List<string>();

            string v = $"*************************************************************************************************************";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*       Metrics for Multi-class Classification model - {description} - Test Data ";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*------------------------------------------------------------------------------------------------------------";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*       LogLoss:          {testMetrics.LogLoss:#.###}";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}";
            reportVals.Add(v);
            // Console.WriteLine(v);

            v = $"*************************************************************************************************************";
            reportVals.Add(v);
            // Console.WriteLine(v);

            string path = modelConfig.path + $"/Results/eval_{abbreviation}{k}.zip";
            File.WriteAllLines(path, reportVals);
        }
    }

    public class Configuration
    {
        public MLContext mlContext;        
        public IDataView testDataView;
        public IEstimator<ITransformer> pipeline;
        public ITransformer model;
        public string path;
    }
}
