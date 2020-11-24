# Preparación
import os
import numpy as np
import pandas as pd


class SisFALL():

  def __init__(self, path, s3Proxy):
    self.pathname = path
    self.data = None
    self.s3 = s3Proxy


  def convert_to_accel(self, df):
    #from mates import module

    df['X'] = df['ax'] * 32 / 8192.0 # ax*2* scale /2^resolution
    df['Y'] = df['ay'] * 32 / 8192.0
    df['Z'] = df['az'] * 32 / 8192.0
    #df['asm'] = df.apply(lambda row: module(row.ax, row.ay, row.az), axis = 1)
    df['t'] = df.index * 0.005 #freq = 200Hz
    return df

  def labels_from_path(self, pathname):
    label, ext = os.path.splitext(os.path.basename(pathname))
    activity, subject, trial = label.split("_")
    return activity, subject, trial

  def create_accel_dataframe(self, pathname):
    df_accel = pd.read_csv(pathname, usecols=[1,2,3], names={'ax': [1], 'ay':[2], 'az':[3]})
    return df_accel

  def generate_df(self):
    txtfiles = [os.path.join(root, name) for root, dirs, files in os.walk(self.pathname) for name in files if name.endswith((".txt"))]

    for filepath in txtfiles:
      ad = self.create_accel_dataframe(filepath)
      activity, subject, trial = labels_from_path(filepath)
      isFall = activity.startswith("F")
      #print(f"{activity} ({isFall}) - {subject} - {trial}")
      #data = convert_to_accel(ad)
      sessions.append({
        "activity" : activity,
        "fall" : isFall,
        "uid" : subject,
        "samplingRate": 200,
        "data" : convert_to_accel(ad)
      })

      self.data = pd.DataFrame(sessions)
      return self.data

  def save(self):
    self.s3.writeDFToS3(self.data, "sisfall_corpus.df")

  def load(self):
    if self.data is None:
      self.data = self.s3.readDFFromS3("sisfall_corpus.df")
    return self.data
