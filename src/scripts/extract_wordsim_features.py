import os

from noisy_ml.data.wordsim import extract_features

def main():
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  extract_features(data_dir)


if __name__ == '__main__':
  main()
