'''
The MIT License (MIT)
Copyright (c) 2017 Pratik Solanke
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Demonstrating the working of gradient descent
The code aims to fit a linear model to the UFO dataset by,
finding the values of m and b using gradient descent
Error is calculated using SSE

Training :  python3 GradientDescent.py --train --csv_file=scrubbed.csv
            python3 GradientDescent.py --train --csv_file=scrubbed.csv --num_iterations=2000 --learning_rate=0.0032 --consts_file='data.json'
Prediction: python3 GradientDescent.py --x=20
            python3 GradientDescent.py --x=20 --model_file='data.json'

'''
#Imports
import sys
import os
import numpy as np
import pandas as pd
import argparse
import json
import io
import matplotlib.pyplot as plt
import time

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

label_list = ['']
conts_list = ['']
ENCODING_FORMAT = 'utf8'
TIMESTAMP = str(int(time.time()))
DATASET_GRAPH_NAME = 'points_'+TIMESTAMP+'.png'
ERROR_GRAPH_NAME = 'error_'+TIMESTAMP+'.png'
CONSTS_GRAPH_NAME = 'constants_'+TIMESTAMP+'.png'

# Takes values of b, m and the output file as arguments 
# Saves these b, m and label_list to the output file,
# in json format
def save_to_file(b, m, consts_file):
    print('Saving Constants to ' + consts_file)
    if b == None or m == None:
        print('Data object is null \n Cannot store constants to file')
        return
    with io.open(consts_file, 'w', encoding=ENCODING_FORMAT) as outfile:
        data = {'b' : b, 'm' : m, 'label_list' : label_list}
        str_ = json.dumps(data, indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

# Adds labels not in the list to the label_list
def add_to_list(label):
    if(label not in label_list):
        label_list.append(label)
    return label_list.index(label)

def get_index(label):
    return label_list.index(label)

# Reads the json file where the consts and label_list are stored
def read_from_file(consts_file):
    print('Retrievin Constants from ' + consts_file)
    with io.open(consts_file, 'r', encoding=ENCODING_FORMAT) as infile:
        data_loaded = json.load(infile)
    return data_loaded['b'], data_loaded['m'], data_loaded['label_list']

# y = mx + b
# m is slope, b is y-intercept
# calculates the mean square error
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
# Reads data from CSV file,
# removes rows containing any number of missing vlues,
# returns the shape and hour of day(from datetime) as dataframe
def read_UFO_csv(filename):
    points = pd.read_csv(filename)
    points = points.dropna()
    df = pd.DataFrame()
    hour = []
    shape = []
    for index, row in points.iterrows():
        hour.append(int(row['datetime'][-5:-3]))
        shape.append(add_to_list(row['shape']))
    plt.plot(hour, shape, 'ro')
    plt.xlabel('Hours')
    plt.ylabel('Shape')
    plt.savefig(DATASET_GRAPH_NAME)
    plt.close()
    df['hours'] = hour
    df['shape'] = shape

    return df

def main(args):
    if args.train:
        if not(os.path.isfile(args.csv_file)):
            print('No CSV file found.\nExiting...')
            sys.exit(1)
        train(args.csv_file, args.num_iterations, args.learning_rate, args.consts_file)
    else:
        if args.x == None:
            print('Value of x not found. Try --help option')
            sys.exit(1)
        predict(args.x, args.model_file)

# Training for linear model using gradient descent
def train(csv_file, num_iterations, learning_rate, consts_file):
    df = read_UFO_csv(csv_file)
    consts_list = []
    b = 0 # initial y-intercept guess
    m = 0 # initial slope guess
    initial_b = b
    initial_m = m 
    intial_error = compute_error_for_line_given_points(b, m, df.values)
    print('Running...')
    for i in range(num_iterations):
        b, m = step_gradient(b, m, df.values , learning_rate)
        error = compute_error_for_line_given_points(b, m, df.values)
        consts_list.append([b,m,error])
        print('Step : ' + str(i) + '\tb : ' + str(b) + '\tm : ' + str(m) + '\terror: ' + str(error))
    plt.plot(range(len(consts_list)), [row[2] for row in consts_list], 'r')
    plt.xlabel('Hours')
    plt.ylabel('Shape')
    plt.savefig(ERROR_GRAPH_NAME)
    plt.close()
    
    plt.subplot(211)
    plt.plot(range(len(consts_list)), [row[1] for row in consts_list], 'r')
    plt.subplot(212)
    plt.plot(range(len(consts_list)), [row[0] for row in consts_list], 'g')
    # plt.plot(range(len(consts_list)), [row[1] for row in consts_list], 'r', range(len(consts_list)), [row[0] for row in consts_list], 'g')
    # plt.xlabel('Hours')
    # plt.ylabel('Shape')
    plt.savefig(CONSTS_GRAPH_NAME)
    plt.close()
    print('Initial values were b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, intial_error))
    print('Final values are b = {0}, m = {1}, error = {2}'.format(b, m, compute_error_for_line_given_points(b, m, df.values)))

    save_to_file(b, m, consts_file)

#Predict using linear model
def predict(x, in_file):
    b, m, label_list = read_from_file(in_file)
    print('b : ' + str(b) + '\tm : ' + str(m))
    y = round(m * x + b)
    print('Predicted Shape: ' + label_list[y])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, help='Path(s) of the csv file')
    parser.add_argument('--num_iterations', type=int, default=500, help='Number of iteration to run')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train', help='Training Option', action='store_true')
    parser.add_argument('--consts_file', type=str, default='final-consts.json', help='Output file where b, m from gradient descent is stored')
    parser.add_argument('--model_file', type=str, default='final-consts.json', help='File to load b, m from for prediction')
    parser.add_argument('--x', type=int, help='Hour of the day in 24hr format')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))