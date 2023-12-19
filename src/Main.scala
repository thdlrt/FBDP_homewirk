import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import spire.random.Random.int

object Main {
  def main(args: Array[String]) {
    //task1()
    //task2()
    task3()
  }
  def task1(): Unit = {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val sc = new SparkContext(conf)
    val fileRDD = sc.textFile("C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\application_data.csv")
    val header = fileRDD.first()
    val dataRDD = fileRDD.filter(row => row != header)
    val data = dataRDD.map{line =>
      val parts = line.split(",")
      val difference = parts(7).toDouble - parts(6).toDouble
      (parts, difference)}
    //子任务1
    val loanAmounts = data.map(x => x._1(7).toDouble)
    def getRange(amount: Double): (Double, Double) = {
      val lower = (amount / 10000).toInt * 10000
      val upper = lower + 10000
      (lower, upper)
    }
    val rangeCounts = loanAmounts.map(amount => (getRange(amount), 1)).reduceByKey(_ + _)
    val sortedConts = rangeCounts.sortBy(_._1._1)
    val outputPath = "C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task1_1"
    sortedConts.saveAsTextFile(outputPath)
    //子任务2
    val res= data.sortBy(_._2,ascending = true)
    val max5 = res.sortBy(x => x._2, ascending = false).take(10)
    val min5 = res.take(10)
    val max5RDD = sc.parallelize(max5)
    val min5RDD = sc.parallelize(min5)
    val res2 = max5RDD.union(min5RDD)
    val res3 = res2.map(t => t._1.mkString(","))
    val outputPath2 = "C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task1_2"
    res3.saveAsTextFile(outputPath2)
    try Thread.sleep(100000)
    catch {
      case e: InterruptedException =>
        throw new RuntimeException(e)
    }
    sc.stop()
  }

  def task2(): Unit = {
    val spark = SparkSession.builder.appName("Spark").master("local").getOrCreate()
    val df = spark.read.option("header", "true").option("inferSchema", "true").csv("C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\application_data.csv")
    val dfM = df.filter(df("CODE_GENDER")==="M")
    val sum = dfM.count()
    val dfG = dfM.groupBy("CNT_CHILDREN").count()
    val res = dfG.select(dfG("CNT_CHILDREN"), (dfG("count")/sum).as("count"))
    val path = "C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task2_1"
    res.write.csv(path)
    val dfI = df.select(df("SK_ID_CURR"),(df("AMT_INCOME_TOTAL")/(-df("DAYS_BIRTH"))).as("avg_income"))
    val dfB = dfI.filter(dfI("avg_income")>1)
    val res2 = dfB.sort(dfB("avg_income").desc)
    val path2 = "C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task2_2"
    res2.write.csv(path2)
    try Thread.sleep(100000)
    catch {
      case e: InterruptedException =>
        throw new RuntimeException(e)
    }
    spark.stop()
  }

  def task3(): Unit = {
    val spark = SparkSession.builder.appName("Spark").master("local").getOrCreate()
    val df = spark.read.option("header", "true").option("inferSchema", "true").csv("C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\application_data.csv")
    val assembler = new VectorAssembler().setInputCols(Array(
      "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
      "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
      "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "FLAG_MOBIL",
      "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE",
      "FLAG_PHONE", "FLAG_EMAIL", "REGION_RATING_CLIENT",
      "REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START",
      "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
      "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
      "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
      "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
      "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
      "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3",
      "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
      "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9",
      "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12",
      "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15",
      "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18",
      "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"
    )).setOutputCol("features").setHandleInvalid("skip")
    val assemblerDf = assembler.transform(df)
    val Array(trainingData, testData) = assemblerDf.randomSplit(Array(0.8, 0.2))
    val classifier1 = new org.apache.spark.ml.classification.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("TARGET")
      .setFeaturesCol("features")
    val model1 = classifier1.fit(trainingData)
    val predictions1 = model1.transform(testData)
    val evaluator1 = new org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator()
      .setLabelCol("TARGET")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy1 = evaluator1.evaluate(predictions1)
    val classifier2 = new org.apache.spark.ml.classification.DecisionTreeClassifier()
      .setLabelCol("TARGET")
      .setFeaturesCol("features")
    val model2 = classifier2.fit(trainingData)
    val predictions2 = model2.transform(testData)
    val evaluator2 = new org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator()
      .setLabelCol("TARGET")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val path1="C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task3_logistic"
    val path2="C:\\Users\\MSI\\OneDrive\\study\\作业\\金融大数据\\spark\\output\\task3_decisiontree"
    val res1 = predictions1.select("SK_ID_CURR", "prediction")
    val res2 = predictions2.select("SK_ID_CURR", "prediction")
    res1.write.csv(path1)
    res2.write.csv(path2)
    val accuracy2 = evaluator2.evaluate(predictions2)
    println("LogisticRegression accuracy = " + accuracy1)
    println("DecisionTreeClassifier accuracy = " + accuracy2)
    try Thread.sleep(100000)
    catch {
      case e: InterruptedException =>
        throw new RuntimeException(e)
    }
    spark.stop()
  }
}
