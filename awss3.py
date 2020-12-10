import boto3
import pickle
import pandas as pandas
import io

class S3proxy():
  def __init__(self, bucket, key, pwd):
    self.BUCKET_NAME = bucket

    self.s3 = boto3.resource('s3',
                    aws_access_key_id = key,
                    aws_secret_access_key = pwd)
    self.s3_bucket = self.s3.Bucket(bucket)


  def writeFileToS3(self, content, filename):
    self.s3.Object(self.BUCKET_NAME, filename).put(Body=content)

  def writeDFToS3(self, ndf, filename):
    pickle_buffer = io.BytesIO()
    pickle.dump(ndf, pickle_buffer)
    self.writeFileToS3(pickle_buffer.getvalue(), filename)

  def readFileFromS3(self, filename):
    return self.s3.Object(self.BUCKET_NAME, filename).get()['Body'].read()

  def readDFFromS3(self, filename):
    return pickle.load(io.BytesIO(self.readFileFromS3(filename)))


  def readFile(self, filename):
    return self.readFileFromS3(filename)
  def writeFile(self, content, filename):
    return self.writeFileToS3(content, filename)

  def readDF(self, filename):
    return self.readDFFromS3(filename)
  def writeDF(self, ndf, filename):
    return self.writeDFToS3(ndf, filename)
