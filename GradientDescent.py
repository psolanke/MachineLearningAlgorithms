import sys
import numpy as np
import pandas as pd
import argparse

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # totalError += (y - (m * x + b)) ** 2
        print(x + '\t' + y)
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
        hour.append(row['datetime'][-5:-3])
        shape.append(row['shape'])

    df['hours'] = hour
    df['shape'] = shape
    return df

def main(args):
    df = read_UFO_csv(args.csv_file)
    learning_rate = 0.0001
    b = 0 # initial y-intercept guess
    m = 0 # initial slope guess
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(b, m, compute_error_for_line_given_points(b, m, df.values)))
    print("Running...")
    for i in range(args.num_iterations):
        b, m = step_gradient(b, m, df.values , learning_rate)
        print("b : " + b + "\tm : " + m)
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(b, m, compute_error_for_line_given_points(b, m, df.values)))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='Path(s) of the csv file')
    parser.add_argument('--num_iterations', type=int, default=1000)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))