// Основная информация 
name := "LinearRegressionComparison"
version := "1.0.0"
scalaVersion := "2.12.15"  

// настройка
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0",
  "org.scalanlp" %% "breeze" % "2.1.0",
  "org.scalatest" %% "scalatest" % "3.2.15" % Test
)


