import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import breeze.linalg.DenseVector
import org.apache.spark.ml.regression.LinearRegressionModel

class LinearRegressionComparisonTest extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var sparkCoefficients: Array[Double] = _
  private var sparkIntercept: Double = _
  private var manualCoefficients: DenseVector[Double] = _
  private var manualIntercept: Double = _
  
  private val tolerance = 0.1
  
  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("LinearRegressionTest")
      .master("local[2]")
      .getOrCreate()
      
    val (sparkModel, manualWeights, manualB) = LinearRegressionComparison.run(spark)
    
    sparkCoefficients = sparkModel.coefficients.toArray
    sparkIntercept = sparkModel.intercept
    manualCoefficients = manualWeights
    manualIntercept = manualB
  }
  
  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }
  
  test("Coefficients Соответствие") {
    sparkCoefficients.zip(manualCoefficients.toArray).foreach { 
      case (spark, manual) =>
        val diff = math.abs(spark - manual)
        assert(diff < tolerance, 
          s"Coefficient слишком большая разница: $spark vs $manual (diff $diff)")
    }
  }
  
  test("Intercept Соответствие") {
    val diff = math.abs(sparkIntercept - manualIntercept)
    assert(diff < tolerance, 
      s"Intercept слишком большая разница: $sparkIntercept vs $manualIntercept (diff $diff)")
  }
  
  test("Manual implementation should capture main relationships") {
    sparkCoefficients.zip(manualCoefficients.toArray).foreach {
      case (spark, manual) =>
        assert((spark * manual) >= 0, 
          s"Coefficient несоответствие : $spark vs $manual")
    }
  }
  
  test("R² Соответствие") {
    val r2 = LinearRegressionComparison.calculateAdjustedR2(spark)
    assert(r2 > 0.8, s"R² слишком низкий: $r2")
  }
  
  test("Manual weights Соответствие") {
    manualCoefficients.foreach { coeff =>
      assert(coeff > -2.0 && coeff < 2.0, s"Coefficient $coeff вне ожидаемого диапазона")
    }
  }
}
