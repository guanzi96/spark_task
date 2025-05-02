import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegressionModel
import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, rand, randn}
import org.apache.spark.ml.regression.LinearRegression
import spark.implicits._

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

    val (sparkModel, manualWeights, manualB) = run(spark)
    
    println("Spark Model Coefficients: " + sparkModel.coefficients)
    println("Spark Intercept: " + sparkModel.intercept)
    println("Manual Weights: " + manualWeights)
    println("Manual Intercept: " + manualB)
    
    spark.stop()
  }

  def run(spark: SparkSession): (LinearRegressionModel, DenseVector[Double], Double) = {
    // Генерировать данные
    val (training, breezeX, breezeY) = generateData(spark)
    
    // Обучение модели Spark
    val sparkModel = trainSparkModel(training)
    
    // Выполнение обучения вручную
    val (manualWeights, manualB) = trainManualImplementation(breezeX, breezeY)
    
    (sparkModel, manualWeights, manualB)
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
    
    // Breeze Формат 
    val (breezeX, breezeY) = convertToBreeze(training)
    
    (training, breezeX, breezeY)
  }

  private def convertToBreeze(training: org.apache.spark.sql.DataFrame) = {
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

  private def trainSparkModel(training: org.apache.spark.sql.DataFrame) = {
    val lr = new LinearRegression()
      .setMaxIter(10000)
      .setRegParam(0.0)
      .setElasticNetParam(0.8)

    lr.fit(training)
  }

  def trainManualImplementation(
    breezeX: DenseMatrix[Double],
    breezeY: DenseVector[Double]
  ): (DenseVector[Double], Double) = {
    var w = DenseVector(0.0, 0.0, 0.0)
    var b = 0.0
    val learningRate = 0.03
    val numIterations = 5000
    val batchSize = 256

    for (iter <- 0 until numIterations) {
      val miniBatchIndices = scala.util.Random.shuffle((0 until numRows).toList).take(batchSize)
      val XBatch = breezeX(miniBatchIndices, ::).toDenseMatrix
      val yBatch = breezeY(miniBatchIndices).toDenseVector

      val predictions = XBatch * w + b
      val errors = predictions - yBatch
      val gradientW = (XBatch.t * errors) * (1.0 / batchSize)
      val gradientB = breeze.linalg.sum(errors) * (1.0 / batchSize)

      w -= learningRate * gradientW
      b -= learningRate * gradientB
    }
    
    (w, b)
  }

  def calculateAdjustedR2(spark: SparkSession): Double = {
    val trainingSummary = trainSparkModel(generateData(spark)._1).summary
    val adjR2 = 1 - (1 - trainingSummary.r2) * (numRows - 1) / (numRows - numFeatures - 1)
    adjR2
  }
}
