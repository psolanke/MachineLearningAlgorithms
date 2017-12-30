import sys
import numpy as np
import pandas as pd
import argparse
import json
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

label_list = ['']
JSON_FILE = 'final-variables.json'
ENCODING_FORMAT = 'utf8'
def save_to_json(b, m):
    print('Saving Variables to ' + JSON_FILE)
    if b == None or a == None:
        print('Data object is null \n Cannot store variables to file')
        return
    with io.open(JSON_FILE, 'w', encoding=ENCODING_FORMAT) as outfile:
        data = {'b' : b, 'm' : m}
        str_ = json.dumps(data, indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

def add_to_list(label):
    if(label not in label_list):
        label_list.append(label)
    return label_list.index(label)

def get_index(label):
    return label_list.index(label)

def read_from_json():
    print('Retrievin Variables from ' + JSON_FILE)
    with io.open(JSON_FILE, 'r', encoding=ENCODING_FORMAT) as infile:
        data_loaded = json.load(infile)
    b = data_loaded['b']
    m = data_loaded['m']
    return b, m

#Predict using linear model
def predict(x):
    b, m = read_from_json()
    print('b : ' + str(b) + '\tm : ' + str(m))

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def read_UFO_csv(filename):
    points = pd.read_csv(filename)
    points = points.dropna()
    df = pd.DataFrame()
    hour = []
    shape = []
    for index, row in points.iterrows():
        hour.append(int(row['datetime'][-5:-3]))
        shape.append(add_to_list(row['shape']))

    df['hours'] = hour
    df['shape'] = shape
    return df

def main(args):


def train(args):
    df = read_UFO_csv(args.csv_file)
    learning_rate = args.learning_rate
    b = 0 # initial y-intercept guess
    m = 0 # initial slope guess
    initial_b = b
    initial_m = m 
    intial_error = compute_error_for_line_given_points(b, m, df.values)
    print("Running...")
    for i in range(args.num_iterations):
        b, m = step_gradient(b, m, df.values , learning_rate)
        print("Step : " + str(i) + "\tb : " + str(b) + "\tm : " + str(m))
    print("Initial values are b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, intial_error))
    print("Final values are b = {0}, m = {1}, error = {2}".format(b, m, compute_error_for_line_given_points(b, m, df.values)))

    save_to_json(b, m)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='Path(s) of the csv file')
    parser.add_argument('--num_iterations', type=int, default=500, help='Number of iteration to run')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--num_iterations', type=int, default=1000)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))