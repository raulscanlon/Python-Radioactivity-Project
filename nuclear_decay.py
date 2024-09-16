# -*- coding: utf-8 -*-
"""
PHYS20161 Final assignment: 79Rb Decay

A sample of 79Sr is collected and then decays. This code determines the
half-life and decay constants of 79Rb and 79Sr.
Figures are generated accordingly to support any findings.

Written by ID 10691947 on 15/12/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.constants import Avogadro

FILE_NAME_1 = 'Nuclear_data_1.csv'
FILE_NAME_2 = 'Nuclear_data_2.csv'
DELIMITER = ','
SKIP_FIRST_LINE = True


def open_file(file_name, skip_first_line=SKIP_FIRST_LINE):
    """
    Opens file, reads data and outputs data in numpy arrays.
    Args:
        file_name: string
    Returns:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
    Raises:
        FileNotFoundError
    """
    x_data = np.array([])
    y_data = np.array([])
    y_uncertainties = np.array([])
    try:
        raw_file_data = open(file_name, 'r')
    except FileNotFoundError:
        print("File '{0:s}' cannot be found.".format(file_name))
        print('Check it is in the correct directory.')
        return x_data, y_data, y_uncertainties
    for line in raw_file_data:
        if skip_first_line:
            skip_first_line = False
        else:
            line_valid, line_data = validate_line(line)
            if line_valid:
                x_data = np.append(x_data, line_data[0])
                y_data = np.append(y_data, line_data[1])
                y_uncertainties = np.append(y_uncertainties, line_data[2])
    raw_file_data.close()
    return x_data, y_data, y_uncertainties


def validate_line(line):
    """
    Validates line by checking if entries are numeric, and whether
    uncertainty is zero.
    Outputs error messages accordingly.
    Args:
        line: string
    Returns:
        bool, if validation has been succesful
        line_floats: numpy array of floats
    """
    line_split = line.split(DELIMITER)
    for entry in line_split:
        if check_numeric(entry) is False:
            print('Line omitted: {0:s}.'.format(line.strip('\n')))
            print('{0:s} is nonnumerical.'.format(entry))
            return False, line_split
    line_floats = np.array([float(line_split[0]), float(line_split[1]),
                            float(line_split[2])])
    if line_floats[2] <= 0:
        print('Line omitted: {0:s}.'.format(line.strip('\n')))
        print('Uncertainty must be greater than zero.')
        return False, line_floats
    return True, line_floats


def check_numeric(entry):
    """
    Checks if entry is numeric.
    Args:
        entry: string
    Returns:
        bool
    Raises:
        ValueError: if entry cannot be cast to float type
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False


def combine_data(big, small_1, small_2):
    """
    Combines validated data from each data file into one numpy array. Also
    removes Nan's from the resultant array.
    Args:
        small_1 : numpy array
        small_2 : numpy array
    Returns:
        big : numpy array
    """
    small_1 = np.transpose(small_1)
    small_2 = np.transpose(small_2)
    big = np.vstack((small_1, small_2))
    big = big[~np.isnan(big).any(axis=1), :]
    big = big[big[:, 0].argsort()]
    return big


def activity_model(time, lambda_sr, lambda_rb):
    """
    Calculates the predicted activity of 79Rb at a given time in seconds based
    on the decay constants for 79Sr and 79Rb.

    Args:
        time : float
        lambda_sr : float
        lambda_rb : float

    Returns:
        activity_predicted : float
    """
    n_sr_initial = 10**-6 * Avogadro
    n_rb = n_sr_initial * (lambda_sr / (lambda_rb - lambda_sr)) * \
        (np.exp(-lambda_sr * time) - np.exp(-lambda_rb * time))
    activity_predicted = lambda_rb * n_rb
    return activity_predicted


def remove_outliers(array):
    """
    Removes outliers from the input array. Uses the model of the 79Rb activity
    and finds the difference between the input data and the expected value.
    If the data is more than 3 times the error on the measurement, the value
    is removed.

    Args:
        array : numpy array
    Returns:
        filtered_data :  numpy array
    """
    filtered_data = []
    lambda_rb_guess = 0.0005
    lambda_sr_guess = 0.005
    for row in array:
        time = row[0]
        activity = row[1]
        error = row[2]
        difference = activity - \
            activity_model(time, lambda_sr_guess, lambda_rb_guess)
        if abs(difference) < 3 * error:
            filtered_data.append(row)
        else:
            print('Line omitted: {}.'.format(', '.join(map(str, row))))
            print('Activity value is out of expected range.')
    filtered_data = np.asarray(filtered_data)
    return filtered_data


def chi_squared(params, time, activity, error):
    """
    Calculates the chi-squared value of the dataset when compared to a
    model that outputs its expected values.

    Args:
        params : numpy array
        time : numpy array
        activity : numpy array
        error : nummpy array
    Returns:
        chi_value : float
    """
    lambda_sr = params[0]
    lambda_rb = params[1]
    activity_predicted = activity_model(time, lambda_sr, lambda_rb)
    chi_value = np.sum((activity - activity_predicted)**2 / error**2)
    return chi_value


def minimised_chi(time, activity, error):
    """
    Minimises the chi squared function in order to find correct values for the
    decay constants of 79Sr and 79Rb.

    Args:
        time : numpy array
        activity : numpy array
        error : numpy array
    Returns
        params : numpy array
    """
    lambda_rb_guess = 0.0005
    lambda_sr_guess = 0.005
    p_0 = (lambda_sr_guess, lambda_rb_guess)
    params = fmin(chi_squared, p_0, args=(time, activity, error))
    return params


def find_halflife(lambda_0):
    """
    Finds the half-life of a sample using the equation below. Converts
    half-like from seconds to minutes.

    Args:
        lambda_0 : float
    Returns:
        t_half : float
    """
    t_half_seconds = (np.log(2))/lambda_0
    t_half = t_half_seconds / 60
    return t_half


def activity_specified(lambda_sr, lambda_rb):
    """
    Finds the activity of the sample after 90 minutes and prints a statement.
    Also asks the user whether they would like to know the activty of the
    sample at another time and outputs the relevant value if requested.

    Args:
        lambda_sr : float
        lambda_rb : float
    Returns:
        activity_90_tbq : float
    """
    activity_90_tbq = activity_model(5400, lambda_sr, lambda_rb) * 10**-12
    print('The activity at 90 minutes is {0:4.1f} TBq.'.format(activity_90_tbq))
    question = input("Would you like to know the actvity of the sample at "
    "another time? If so, type 'yes', otherwise type any other input.")
    question = question.replace(" ", "")
    if question == 'yes':
        question_time = input("After how many seconds would you"
                              " like to know the activity?")
        question_time = question_time.replace(" ", "")
        if question_time.isnumeric() is False:
            print("Please enter numerical values only.")
        if question_time.isnumeric() is True:
            activity_q_tbq = activity_model(float(question_time),
                                            lambda_sr, lambda_rb) * 10**-12
            print("The activity at {0} seconds is {1:4.2f} TBq.".\
            format(question_time, activity_q_tbq))
            return activity_90_tbq


def red_chi(lambda_sr, lambda_rb, data):
    """
    Calculates the chi-squared value for decay constant values found and then
    determines the reduced chi square value and prints.

    Args:
        lambda_sr : float
        lambda_rb : float
        data : numpy array
    Returns:
        reduced_chi_square : float
    """
    decay_values = [lambda_sr, lambda_rb]
    time = data[:, 0]
    activity = data[:, 1]
    error = data[:, 2]
    chi_value = chi_squared(decay_values, time, activity, error)
    reduced_chi_squared = chi_value / ((len(data[:, 1]))-2)
    return reduced_chi_squared


def create_plot(data):
    """
    Plotting routine that plots the activity model against the data points.
    Formats any findings in this code beneath the graphic and outputs as a
    PNG image.
    Args:
        data: numpy array
    Returns:
        minimised_values : tuple
    """
    x_data = data[:, 0]
    y_data = data[:, 1]
    y_uncertainties = data[:, 2]
    minimised_values = minimised_chi(x_data, y_data, y_uncertainties)
    lambda_rb = minimised_values[1]
    lambda_sr = minimised_values[0]
    params = [lambda_sr, lambda_rb]
    reduced_chi_squared = red_chi(lambda_sr, lambda_rb, data)
    rb_time = find_halflife(lambda_rb)
    sr_time = find_halflife(lambda_sr)
    figure = plt.figure(figsize=(8, 6))
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.errorbar(x_data, y_data, yerr=y_uncertainties,
                            fmt='x', color='black')
    axes_main_plot.plot(x_data, activity_model(x_data, lambda_sr, lambda_rb),
                        color='r')
    axes_main_plot.grid(True)
    axes_main_plot.set_title('Activity of sample (Bq) against Time (seconds)'
                             ,fontsize=14)
    axes_main_plot.set_xlabel('Time (seconds)')
    axes_main_plot.set_ylabel('Activity (Bq)')
    axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.2f}'.
                             format(reduced_chi_squared)), (0, 0), (0, -70),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate('Half-life of 79Sr: {0:4.2f} minutes'.\
                            format(sr_time), (1, 0), (-160, -55),\
                            xycoords='axes fraction', va='top',\
                            textcoords='offset points')
    axes_main_plot.annotate('Half-life of 79Rb:{0:4.1f} minutes'.\
                            format(rb_time), (1, 0), (-160, -35),xycoords=\
                            'axes fraction', va='top', textcoords\
                            ='offset points')
    axes_main_plot.annotate(('79Rb Decay Constant = {0:4.6f} s^-1'.format\
                            (params[1])), (0, 0),(0, -35), xycoords='axes \
                            fraction', va='top',textcoords='offset points', \
                            fontsize='10')
    axes_main_plot.annotate(('79Sr Decay Constant = {0:4.5f} s^-1'.format\
                            (params[0])), (0, 0),(0, -55), xycoords='axes \
                            fraction', va='top', textcoords='offset points',\
                            fontsize='10')
    residuals = y_data - activity_model(data[:, 0], lambda_sr, lambda_rb)
    axes_residuals = figure.add_subplot(414)
    axes_residuals.errorbar(x_data, residuals, yerr=y_uncertainties,
                            fmt='x', color='black')
    axes_residuals.plot(x_data, 0 * x_data, color='r')
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14)
    plt.savefig('fit_result.png', dpi=600, transparent=True)
    plt.show()
    return minimised_values


def main():
    """
    Main routine that calls each function in turn.

    Returns:
        data: numpy array

    """
    data_1 = np.array(open_file(FILE_NAME_1))
    data_2 = np.array(open_file(FILE_NAME_2))
    data = np.zeros((0, 3))
    data = combine_data(data, data_1, data_2)
    data[:, 0] = data[:, 0] * 3600
    data[:, 1] = data[:, 1] * 10**12
    data[:, 2] = data[:, 2] * 10**12
    data = remove_outliers(data)
    decay_values = create_plot(data)
    lambda_rb = decay_values[1]
    lambda_sr = decay_values[0]
    activity_specified(lambda_sr, lambda_rb)
    return data


if __name__ == '__main__':
    main()
