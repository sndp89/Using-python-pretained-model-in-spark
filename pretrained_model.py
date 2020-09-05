## First need to make sure xgboost package is installed on all nodes
import xgboost as xgb
### Load trained model as .pkl file
model_xgb =  pickle.load(open('model_xgb.pkl', 'rb'))
## Define your function
gef get_prediction(text):
  df_idf = tfidf_vec.transform(np.array([text]).tolist())
  return (model_xgb.predict(xgb.DMatrix(df_idf.todense())))[0].item()

# import udf package
from pyspark.sql.functions import udf

## Register UDF
sqlContext.udf.register("myPrediction", get_prediction, FloatType())

## you can define it as a UDF
myPrediction = udf(lambda x: get_prediction(x), returnType=FloatType())

# Now you can either add it as a column to a dataframe
df_pred = df.withColumn('predicted', myPrediction('utterance'))
df_pred.createOrReplaceTempView("pred")
