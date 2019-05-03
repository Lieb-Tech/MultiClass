using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MultiClass
{
    public class PredictionTest
    {
        MLContext _mlContext;
        string _basePath;
        List<string[]> testDataSet;
        public PredictionTest(MLContext mlContext, string basePath)
        {
            _mlContext = mlContext;
            _basePath = basePath;
        }

        public void RunPredictions()
        {
            var rawTestData = File.ReadAllLines($"{_basePath}/Data/evaluate_data.tsv");
            testDataSet = rawTestData.Select(z => z.Split('\t')).Skip(1).ToList().Where(z => z.Count() == 5).ToList();

            var results = new List<string>();
            var files = Directory.GetFiles($"{_basePath}/Models", "model*.zip");
            foreach (var file in files.OrderBy(z => z))
            {
                var v = runPred(file);
                if (v.Item1 != -1)
                {
                    results.Add($"{v.Item2} = {((v.Item1 * 1.0) / 1000) * 100} %");
                }
            }
            File.WriteAllLines($"{_basePath}/Results/pred_all.txt", results);
        }

        Tuple<int, string> runPred(string modelPath)
        {            
            var idx1 = modelPath.IndexOf("_") + 1;
            var idx2 = modelPath.IndexOf(".zip");
            var description = modelPath.Substring(idx1, idx2 - idx1);

            Console.WriteLine("Running predictions for : " + description);
            try
            {
                ITransformer loadedModel;
                using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    loadedModel = _mlContext.Model.Load(stream, out var modelInputSchema);
                }

                var _predEngine = _mlContext.Model.CreatePredictionEngine<NewsItem, SectionPrediction>(loadedModel);

                int correct = 0;
                foreach (var t in testDataSet)
                {
                    var result = _predEngine.Predict(new NewsItem()
                    {
                        Title = t[3],
                        Description = t[4],
                    });
                    if (result.Section == t[1])
                        correct++;
                }

                File.WriteAllLines($"{_basePath}/Results/pred_{description}.txt", new List<string>()
            {
                "***************************",
                "Prediction results: " + description,
                "***************************",
                "Test set size: " + testDataSet.Count(),
                "Total correct: " + correct,
                "Perc correct : " + ((correct * 1.0) / testDataSet.Count()) * 100.0 + "%"
            });
                return new Tuple<int, string>(correct, description);
            }
            catch (Exception e)
            {
                return new Tuple<int, string>(-1, "");
            }
        }
    }
}
