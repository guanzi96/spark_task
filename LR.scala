import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegressionModel
import breeze.linalg.{DenseVector, DenseMatrix, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, rand, randn}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame
import scala.util.Random

object LinearRegressionComparison {
  private val numRows = 100000
  private val coefficients = Array(1.5, 0.3, -0.7)
  private val noiseLevel = 0.1
  private val numFeatures = coefficients.length

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("LinearRegressionComparison")
      .master("local[*]")
      .getOrCreate()

    val (sparkModel, manualModel) = run(spark)
    
    spark.stop()
  }

  // параметрический класс
  class ManualParams(
    var learningRate: Double = 0.03,
    var numIterations: Int = 5000,
    var batchSize: Int = 256,
    var regParam: Double = 0.1,
    var elasticNetParam: Double = 0.8
  )

  // модели
  class ManualModel(
    val coefficients: DenseVector[Double],
    val intercept: Double
  ) extends Serializable {
    def predict(features: DenseVector[Double]): Double = {
      coefficients.dot(features) + intercept
    }
  }

  //  класс оценщика
  class ManualEstimator(params: ManualParams = new ManualParams()) extends Serializable {
    def fit(X: DenseMatrix[Double], y: DenseVector[Double]): ManualModel = {
      
      var w = DenseVector.zeros[Double](X.cols)
      var b = 0.0
      
      for (iter <- 0 until params.numIterations) {
        val miniBatchIndices = Random
          .shuffle((0 until X.rows).toList)
          .take(params.batchSize)
        
        val XBatch = X(miniBatchIndices, ::).toDenseMatrix
        val yBatch = y(miniBatchIndices).toDenseVector
        
        val predictions = XBatch * w + b
        val errors = predictions - yBatch
        
        // Вычисление градиента
        val gradientW = (XBatch.t * errors) * (1.0 / params.batchSize)
        val gradientB = sum(errors) * (1.0 / params.batchSize)
        
        // регуляризации
        val l2Reg = w * (params.regParam * params.elasticNetParam)
        
        w -= params.learningRate * (gradientW + l2Reg)
        b -= params.learningRate * gradientB
      }
      
      new ManualModel(w, b)
    }
  }

  def run(spark: SparkSession): (LinearRegressionModel, ManualModel) = {
    val (training, breezeX, breezeY) = generateData(spark)
    
    // Обучение модели Spark
    val sparkModel = trainSparkModel(training)
    
    // Выполнение обучения 
    val manualParams = new ManualParams(
      learningRate = 0.03,
      numIterations = 5000,
      regParam = 0.1,
      elasticNetParam = 0.8
    )
    
    val manualEstimator = new ManualEstimator(manualParams)
    val manualModel = manualEstimator.fit(breezeX, breezeY)
    
    (sparkModel, manualModel)
  }

  private def generateData(spark: SparkSession) = {
    
    val rawData = spark.range(numRows)
      .withColumn("c1", rand(seed=42) * 10)
      .withColumn("c2", rand(seed=84) * 5)
      .withColumn("c3", rand(seed=168) * 2)
      .drop("id")

    val dataWithLabel = rawData.withColumn("label",
      col("c1") * coefficients(0) + 
      col("c2") * coefficients(1) + 
      col("c3") * coefficients(2) + 
      randn(seed=42) * noiseLevel
    )

    val assembler = new VectorAssembler()
      .setInputCols(Array("c1", "c2", "c3"))
      .setOutputCol("features")

    val training = assembler.transform(dataWithLabel).select("features", "label")
    
    //  Breeze Формат 
    val (breezeX, breezeY) = convertToBreeze(training)
    
    (training, breezeX, breezeY)
  }

  private def convertToBreeze(training: DataFrame) = {
    val rows = training.collect()
    val breezeX = DenseMatrix.zeros[Double](numRows, numFeatures)
    val breezeY = DenseVector.zeros[Double](numRows)

    rows.zipWithIndex.foreach { case (row, idx) =>
      val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features").toArray
      breezeX(idx, ::) := DenseVector(features).t
      breezeY(idx) = row.getAs[Double]("label")
    }
    
    (breezeX, breezeY)
  }

  private def trainSparkModel(training: DataFrame) = {
    val lr = new LinearRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.8)

    lr.fit(training)
  }

  def calculateAdjustedR2(spark: SparkSession): Double = {
    val trainingSummary = trainSparkModel(generateData(spark)._1).summary
    val adjR2 = 1 - (1 - trainingSummary.r2) * (numRows - 1) / (numRows - numFeatures - 1)
    adjR2
  }
}
