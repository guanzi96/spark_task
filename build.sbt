// 项目基本信息
name := "LinearRegressionComparison"
version := "1.0.0"
scalaVersion := "2.12.15"  // 需要与Spark 3.3.0兼容的版本

// 依赖库配置
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0",
  "org.scalanlp" %% "breeze" % "2.1.0",
  "org.scalatest" %% "scalatest" % "3.2.15" % Test
)


