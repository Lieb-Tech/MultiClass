using Microsoft.Azure.Documents.Client;
using Microsoft.Azure.Documents.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultiClass
{
    public class FileBuilder
    {
        private int testAmount = 500;
        private int evaluateAmount = 1000;
        CosmosDB cdb;
        public FileBuilder(CosmosDB db)
        {
            cdb = db;
        }

        public async Task BuildModels(string path)
        {            
            Console.WriteLine("Getting data count");
            int reservedAmount = evaluateAmount + testAmount;

            // set Cosmos options for batching
            FeedOptions queryOptions = new FeedOptions { MaxItemCount = 1000, EnableCrossPartitionQuery = true };
            var uri = UriFactory.CreateDocumentCollectionUri("liebfeeds", "newsfeed");
            
            var qry = "SELECT VALUE Count(1) from c";
            var countQry = await cdb.GetDocumentQuery("newsfeed", qry).AsDocumentQuery().ExecuteNextAsync();
            var cnt = countQry.First();

            Console.WriteLine($"Total entry count: {cnt} ; adjusted: {cnt - reservedAmount}");

            // remove testFile from count
            cnt -= reservedAmount;

            // get what should be max file #
            var maxFiles = (int)(cnt / 5000) * 5;
            string modelDataPath = $"{path}\\Data\\train_data{maxFiles}.tsv";

            // if already exists, no need to recreate
            if (File.Exists(modelDataPath))
            {
                Console.WriteLine("Training files already exist");
            }
            else
            {
                Console.WriteLine("Loading data");
                // get the data
                qry = "SELECT c.id, c.title, c.description, c.partionKey, c.siteSection FROM c WHERE c.siteSection <> null";
                var docQry = cdb.GetDocumentQuery("newsfeed", qry, queryOptions).AsDocumentQuery();

                // store results
                var vals = new List<CosmosNewsItem>();

                // batch load data
                while (docQry.HasMoreResults)
                {
                    var z = await docQry.ExecuteNextAsync<CosmosNewsItem>();
                    vals.AddRange(z
                        .Where(v => !string.IsNullOrWhiteSpace(v.title) && !string.IsNullOrWhiteSpace(v.title))
                        .Select(v => v)
                        .ToList()
                    );
                }

                Console.WriteLine("Data loaded: " + vals.Count());
                // build test set from first set of records (so they never change, from run to run)
                
                string testDataPath = $"{path}\\Data\\test_data.tsv";
                if (!File.Exists(testDataPath))
                {
                    File.WriteAllText(testDataPath, "ID\tFeed\tSiteSection\tTitle\tDescription\r\n");
                    File.AppendAllLines(testDataPath, vals
                        .Take(testAmount)
                        .Select(v => $"{v.id}\t{v.partionKey}\t{v.siteSection}\t{v.title}\t{v.description}")
                        .ToList());
                }

                testDataPath = $"{path}\\Data\\evaluate_data.tsv";
                if (!File.Exists(testDataPath))
                {
                    File.WriteAllText(testDataPath, "ID\tFeed\tSiteSection\tTitle\tDescription\r\n");
                    File.AppendAllLines(testDataPath, vals
                        .Skip(testAmount)
                        .Take(evaluateAmount)
                        .Select(v => $"{v.id}\t{v.partionKey}\t{v.siteSection}\t{v.title}\t{v.description}")
                        .ToList());
                }

                // now build the training data files, in batches of 5k,
                int modelAmount = 5000;
                while (modelAmount + reservedAmount < vals.Count())
                {
                    int k = modelAmount / 1000;
                    modelDataPath = $"{path}\\Data\\train_data{k}.tsv";

                    if (!File.Exists(modelDataPath))
                    {
                        Console.WriteLine($"Creating file {k}");
                        File.WriteAllText(modelDataPath, "ID\tFeed\tSiteSection\tTitle\tDescription\r\n");
                        File.AppendAllLines(modelDataPath, vals
                            .Skip(reservedAmount).Take(modelAmount)
                            .Select(v => $"{v.id}\t{v.partionKey}\t{v.siteSection}\t{v.title}\t{v.description}")
                            .ToList());
                    }
                    modelAmount += 5000;
                }
            }

            Console.WriteLine("Training file build compeleted");
        }
    }
}
