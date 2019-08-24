import java.util.List;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

// http://www.stepbystepcoder.com/using-spark-for-anomaly-fraud-detection-k-means-clustering/

public class AnomalyApp {

    public static final String DataPath = "C:\\frank\\workspace\\SparkClusteringAnomalyApp\\data.txt";

    public static void main(String[] args) 
    {
        // Create Spark Context
        JavaSparkContext jsc = new JavaSparkContext("local", "Anomaly Detection");

        // read data from file and cache the result dataRDD
        JavaRDD<Vector> dataRDD = jsc.textFile(DataPath).map(new Function<String, Vector>() {
            public Vector call(String line) throws Exception {
                String[] dataArr = line.split(",");

                // take first values to create "simple" clusters ( easy to debug/learn )
                double[] values = new double[1];
                for (int i = 0; i < 1; i++) 
                {
                    values[i] = Double.parseDouble(dataArr[i]);
                	System.out.print(values[i]+" ");
                }
                System.out.println("");
                return Vectors.dense(values);
            }
        }).cache();
        
       // normalize the data (make sure all double values)
       JavaDoubleRDD firstColumn = dataRDD.mapToDouble(new DoubleFunction<Vector>() {
            public double call(Vector t) throws Exception {
                return t.apply(0);
            }
        });
        
       // compute mean & stdev of the data loaded -- will be used later on to filter data
        final double mean = firstColumn.mean();
        final double stdev = firstColumn.stdev();
        
        System.out.println("Meaning value : " + mean + " Standard deviation : " + stdev + " Max : " + firstColumn.max() + " Min : " + firstColumn.min());
        
        // filter data : we take only "clean" data to train the model 
        JavaRDD<Vector> filteredKddRDD = dataRDD.filter(new Function<Vector, Boolean>() {

            public Boolean call(Vector v1) throws Exception {
                double src_bytes = v1.apply(0);
                if (src_bytes > (mean - 2 * stdev) && src_bytes < (mean + 2 * stdev)) {
                    return true;
                }
                return false;
            }
        }).cache();        

        // create K-Means model
        final int numClusters = 3;
        final int numIterations = 10;

        // FP: used the filtered model for the training, full data set ( dataRDD) later on to predict values 
        final KMeansModel clusters = KMeans.train(filteredKddRDD.rdd(), numClusters, numIterations);
        

        // Take cluster centers
        Vector[] clusterCenters = clusters.clusterCenters();        
        
        // FP : Test 1 
        // uses the full data ( so including 'dummy' values not filtered as for the training model
        
        // calculate the distance between points and center of identified cluster for the value
 		JavaPairRDD<Double,Vector> result1 = dataRDD.mapToPair(new PairFunction<Vector, Double,Vector>() {
            public Tuple2<Double, Vector> call(Vector point) throws Exception {
                int centroidIndex = clusters.predict(point);  //find centroid index

                Vector centroid = clusterCenters[centroidIndex]; //get cluster center (centroid) for given point
                
                //calculate distance
                double preDis = 0;
                for(int i = 0 ; i < centroid.size() ; i ++){
                    preDis = Math.pow((centroid.apply(i) - point.apply(i)), 2);
                    
                }
                double distance = Math.sqrt(preDis);
                System.out.println ("point : "+point+" centroidIndex "+centroidIndex+" centroid "+ centroid +" preDis "+preDis+" distance "+distance);
                return new Tuple2<Double, Vector>(distance, point);
            }
        });   
        
        // list  top 10 suspicious points .. probably we can even better determine if suspicious or not based on distance value
 		List<Tuple2<Double, Vector>> result = result1.sortByKey(false).take(10);   // sort by distance descending 
       	for(Tuple2<Double, Vector> tuple : result){
       		System.out.println("suspicious " + tuple);
       	} 
       	
        
       	// FP: Test 2 - Use predict only  ... a very more basic way to compute distance ..
       	System.out.println("Cluster centers:");
        for (Vector center : clusterCenters) {
          System.out.println(" " + center);
        }
       	
       	Double value = 15.8;
       	int clusterIndex = clusters.predict( Vectors.dense( value) );
       	double x = clusterCenters[clusterIndex].apply(0);
       	System.out.println (" clusterIndex for "+value + " = "+ clusterIndex+" 'simple' distance from cluster center "+(x-value));

       	value = 55.8;
       	clusterIndex = clusters.predict( Vectors.dense( value) );
       	x = clusterCenters[clusterIndex].apply(0);
       	System.out.println (" clusterIndex for "+value + " = "+ clusterIndex+" 'simple' distance from cluster center "+(x-value));

       	value = 330.0;
       	clusterIndex = clusters.predict( Vectors.dense( value) );
       	x = clusterCenters[clusterIndex].apply(0);
       	System.out.println (" clusterIndex for "+value + " = "+ clusterIndex+" 'simple' distance from cluster center "+(x-value));
    }
}