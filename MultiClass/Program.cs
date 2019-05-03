using Microsoft.ML;
using System;
using System.IO;

namespace MultiClass
{
    class Program
    {
        private static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static MLContext _mlContext;    
                
        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            CosmosDB client = new CosmosDB();
            client.OpenConnection();            

            var mb = new FileBuilder(client);

            var basePath = Path.Combine(_appPath, "..", "..", "..");
            mb.BuildModels(basePath).Wait();

            var mt = new ModelTrainer(_mlContext, basePath);
            mt.BuildAndTrain();

            var pt = new PredictionTest(_mlContext, basePath);
            pt.RunPredictions();
        }        
    }
}
