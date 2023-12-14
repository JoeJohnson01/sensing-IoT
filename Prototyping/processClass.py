import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz
from ahrs.filters import Madgwick
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error


class dataProcessor:
    """
    This class processes the data from the sensor and splits it into reps.

    A rep consists of three phases:
    1. The initial phase where the sensor is moving (e.g. eccentric)
    2. The stationary phase where the sensor is not moving (e.g. isometric)
    3. The final phase where the sensor is moving again (e.g. concentric)

    It takes in a pandas dataframe with the following columns:
    gx, gy, gz, ax, ay, az, mx, my, mz, unix_time_ms

    Stationary threshold adjusts how still the sensor has to be to be considered stationary.

    Time limit adjusts the minimum time length of a phase, in milliseconds.
    """

    def __init__(self, stationaryThreshold=10000, timeLimit=10):
        self.timeLimit = timeLimit
        self.stationaryThreshold = stationaryThreshold
        self.callibrationData = []

    def formatData(self, data):
        """
        This function formats the data into a pandas dataframe with column names.

        Used to ensure that the data is in the correct format.
        """
        data.columns = ['gx', 'gy', 'gz', 'ax', 'ay',
                        'az', 'mx', 'my', 'mz', 'unix_time_ms']
        data['datetime'] = pd.to_datetime(data['unix_time_ms'], unit='ms')
        data = self.G2MS2(data)
        data = data.groupby('datetime').mean()
        data = data.set_index('unix_time_ms')
        data = self.filter_noise(data)
        data = self.handle_outliers(data)
        data = self.transformToUpright(data)
        data = self.removeGravity(data)
        compound_mag = self.compoundMagnitude(data)  # pass data as argument
        data['compound_magnitude'] = compound_mag
        return data

    def G2MS2(self, data):
        """
        This function converts the accelerometer data from G to m/s^2.
        """
        data['ax'] = data['ax'] * 9.81
        data['ay'] = data['ay'] * 9.81
        data['az'] = data['az'] * 9.81
        return data

    def filter_noise(self, df, fc=0.1):
        b, a = butter(10, fc, btype='low')
        df.iloc[:, 3:6] = filtfilt(b, a, df.iloc[:, 3:6], axis=0)
        df.iloc[:, 0:3] = filtfilt(b, a, df.iloc[:, 0:3], axis=0)
        return df

    def handle_outliers(self, df, threshold=3):
        mean = df.iloc[:, 3:6].mean()
        std_dev = df.iloc[:, 3:6].std()
        z_scores = (df.iloc[:, 3:6] - mean) / std_dev
        return df[(z_scores < threshold).all(axis=1)]

    def transformToUpright(self, data):
        """
        This function transforms the data so that the readings are rotated to be upright at the start of the exercise.

        This means that the sensor does not have to be held in a specific orientation when performing the exercise.

        Takes a pandas dataframe with columns:
        gx, gy, gz, ax, ay, az, mx, my, mz, and returns the same but with the data transformed.
        index is the unix time in milliseconds.

        Gyroscope data is measured in degrees per second.
        Accelerometer data is measured in m/s^2.

        As the sensor starts at rest, the accelerometer data should be [0, 0, 9.81] at the start of the exercise.
        """
        # Calculate the magnitude of the accelerometer data
        data['a_magnitude'] = np.sqrt(
            data['ax']**2 + data['ay']**2 + data['az']**2)

        # Find the index of the row where the accelerometer magnitude is closest to 9.81 m/s^2
        upright_idx = (data['a_magnitude'] - 9.81).abs().idxmin()
        acc_reading = data.loc[upright_idx, ['ax', 'ay', 'az']]

        # Calculate the rotation needed
        current_orientation = acc_reading / np.linalg.norm(acc_reading)
        desired_orientation = np.array([0, 0, 1])
        rotation = R.align_vectors([desired_orientation], [
                                   current_orientation])[0]

        # Apply the rotation to the accelerometer, gyroscope, and magnetometer data
        for sensor_data in [['ax', 'ay', 'az'], ['gx', 'gy', 'gz'], ['mx', 'my', 'mz']]:
            data[sensor_data] = rotation.apply(data[sensor_data])

        # Drop the auxiliary magnitude column
        data.drop('a_magnitude', axis=1, inplace=True)

        return data

    def removeGravity(self, data):
        """
        Removes the gravity component from the accelerometer data.

        It is assumed that at the start of the measurement, the sensor is stationary and thus only measures the gravity.
        The initial readings of the accelerometer are used to estimate the gravity vector.
        """

        # Estimate the gravity vector from the initial stationary period
        # You can adjust the number of initial rows used for this estimation
        initial_readings = data.iloc[:10]
        gravity_vector = np.mean(initial_readings[['ax', 'ay', 'az']], axis=0)

        # Normalize the gravity vector to match the standard gravity magnitude (9.81 m/s^2)
        gravity_vector = 9.81 * gravity_vector / np.linalg.norm(gravity_vector)

        # Initialize a dataframe to store the adjusted accelerometer data
        adjusted_data = data.copy()

        # Subtract the estimated gravity vector from each accelerometer reading
        for index, row in data.iterrows():
            acc_reading = np.array([row['ax'], row['ay'], row['az']])
            adjusted_acc = acc_reading - gravity_vector

            # Update the adjusted data
            adjusted_data.at[index, 'ax'] = adjusted_acc[0]
            adjusted_data.at[index, 'ay'] = adjusted_acc[1]
            adjusted_data.at[index, 'az'] = adjusted_acc[2]

        return adjusted_data

    def compoundMagnitude(self, data):
        """
        Calculates an arbitrary measure called compound magnitude.

        This is the product of the magnitude of the accelerometer, gyroscope and magnetometer.

        Used to determine whether the sensor is moving or not.
        """
        acc_mag = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
        mag_mag = np.sqrt(data['mx']**2 + data['my']**2 + data['mz']**2)
        gyro_mag = np.sqrt(data['gx']**2 + data['gy']**2 + data['gz']**2)

        # Calculate the rate of change for the magnetic field magnitude
        mag_rate_of_change = mag_mag.diff().fillna(0)

        # scale all the data to be between 0 and 100
        acc_mag = (acc_mag - acc_mag.min()) / \
            (acc_mag.max() - acc_mag.min()) * 100
        mag_rate_of_change = (mag_rate_of_change - mag_rate_of_change.min()) / \
            (mag_rate_of_change.max() - mag_rate_of_change.min()) * 100

        gyro_mag = (gyro_mag - gyro_mag.min()) / \
            (gyro_mag.max() - gyro_mag.min()) * 100

        compound_mag = acc_mag * mag_rate_of_change * gyro_mag
        smoothedMag = compound_mag.rolling(window=50).mean()
        return smoothedMag

    def splitDataIntoReps(self, data, plot=False):
        derivativeThreshold = 100
        avgCount = 50
        # calculate a rolling average of the compound magnitude
        avg = data['compound_magnitude'].rolling(
            window=avgCount, center=True).mean()

        # Create another dataframe with the derivative of the rolling average
        derivative = avg.diff().fillna(0)

        # Create a series to identify upward crossings of the positive threshold under the magnitude threshold
        crosses_positive_up = ((derivative.shift(1) < derivativeThreshold) &
                               (derivative >= derivativeThreshold))

        # # Create a series to identify downward crossings of the positive threshold under the magnitude threshold
        # crosses_positive_down = ((derivative.shift(1) >= derivativeThreshold) &
        #                          (derivative < derivativeThreshold))

        # Create a series to identify upward crossings of the negative threshold under the magnitude threshold
        crosses_negative_up = ((derivative.shift(1) < -derivativeThreshold) &
                               (derivative >= -derivativeThreshold))

        # # Create a series to identify downward crossings of the negative threshold under the magnitude threshold
        # crosses_negative_down = ((derivative.shift(1) > -derivativeThreshold) &
        #                          (derivative <= -derivativeThreshold))

        # Combine all series
        # crosses = crosses_positive_up | crosses_positive_down | crosses_negative_up | crosses_negative_down
        crosses = crosses_positive_up | crosses_negative_up

        # remove crossings

        # Split the dataset into a list of dataframes where the derivative crosses the threshold
        split_points = data[crosses].index
        dataframes = [data.iloc[data.index.get_loc(split_point) + 1:data.index.get_loc(
            next_split_point)] for split_point, next_split_point in zip(split_points, split_points[1:])]

        # Remove all dataframes before the first with a max over the magnitude threshold
        for i, df in enumerate(dataframes):
            if df['compound_magnitude'].max() < (avg.max()/4):
                dataframes.pop(i)
            else:
                break

        # Split the dataframes into a list of lists, where each sublist is a rep containing 4 dataframes
        reps = [dataframes[i:i+4] for i in range(0, len(dataframes), 4)]

        # Remove the last dataframe of each rep
        for rep in reps:
            if len(rep) == 4:
                rep.pop()

        # Optionally plot the data
        if plot:
            # plt.figure(figsize=(10, 8))
            plt.figure(figsize=(10, 4))
            # plt.subplot(3, 1, 1)
            # plt.plot(data['compound_magnitude'], label='Compound Magnitude')
            # plt.plot(avg, label='Rolling Average')
            # x axis title
            plt.xlabel('Time')
            # y axis title
            plt.ylabel('Magnitude')
            # tile
            plt.title('Reps and Phases of the Exercise')

            # plot the magnitude threshold in pink
            # plt.plot([data.index[0], data.index[-1]], [
            #     magnitudeThreshold, magnitudeThreshold], '--', color='pink', label='Magnitude Threshold')
            # plt.legend()
            # plt.subplot(3, 1, 2)
            # plt.plot(derivative)
            # # add a dotted green line at y=0, starting at data.index[0] and ending at data.index[-1]
            # plt.plot([data.index[0], data.index[-1]], [
            #     derivativeThreshold, derivativeThreshold], '--', color='green', label='Derivative Upper Threshold')
            # plt.plot([data.index[0], data.index[-1]], [
            #     -derivativeThreshold, -derivativeThreshold], '--', color='red', label='Derivative Lower Threshold')
            # plt.legend()
            # plt.subplot(3, 1, 3)

            # Define a list of colors
            colors = ['b', 'g', 'r']

            for i, rep in enumerate(reps):
                # Use modulo operator to cycle through colors for any length of reps
                rep_color = colors[i % len(colors)]

                for j, phase in enumerate(rep):
                    plt.plot(phase['compound_magnitude'], label='Rep {}, Phase {}'.format(
                        i+1, j+1), color=rep_color)

            for sp in split_points:
                plt.axvline(x=sp, color='black', linestyle='--', alpha=0.7)

            plt.show()

        return reps

    def splitDataIntoReps_v1(self, data, plot=False):
        """
        Populates the set attribute with the processed data.

        This breaks down the data into a list of reps, each of which is a list of phases.

        If plot is True, it will also plot a graph of the compound magnitude and where the phases are.
        """
        threshold = self.stationaryThreshold
        timeLimit = self.timeLimit

        highlight_regions_below = []
        highlight_regions_exceed = []

        current_streaks = {'below': 0, 'exceed': 0}
        start_times = {'below': None, 'exceed': None}
        end_times = {'below': None, 'exceed': None}

        magnitude_data = data['compound_magnitude']

        data_segments = []  # List to hold data within each shaded region

        for index, value in magnitude_data.items():
            condition = 'below' if value <= threshold else 'exceed'
            opposite = 'exceed' if condition == 'below' else 'below'

            current_streaks[condition] += 1
            if start_times[condition] is None:
                start_times[condition] = index
            end_times[condition] = index

            # Condition switched from exceed to below or vice versa
            if current_streaks[opposite] > 0:
                if current_streaks[opposite] >= timeLimit:
                    segment = data.loc[start_times[opposite]:end_times[opposite]]
                    data_segments.append(segment)

                    if opposite == 'below':
                        highlight_regions_below.append(
                            (start_times[opposite], end_times[opposite]))
                    else:
                        highlight_regions_exceed.append(
                            (start_times[opposite], end_times[opposite]))

                # Reset the streaks and times for the opposite condition regardless of whether they were stored
                current_streaks[opposite] = 0
                start_times[opposite] = None
                end_times[opposite] = None

        # After loop: If a streak is still ongoing and is longer than the timeLimit, append it
        if start_times['below'] and current_streaks['below'] >= timeLimit:
            segment = data.loc[start_times['below']:end_times['below']]
            data_segments.append(segment)
            highlight_regions_below.append(
                (start_times['below'], end_times['below']))

        if start_times['exceed'] and current_streaks['exceed'] >= timeLimit:
            segment = self.data.loc[start_times['exceed']:end_times['exceed']]
            data_segments.append(segment)
            highlight_regions_exceed.append(
                (start_times['exceed'], end_times['exceed']))

        # remove any data before the first rep begins

        # calculates the max value of the compound magnitude for the set
        max = 0
        for segment in data_segments:
            if segment['compound_magnitude'].max() > max:
                max = segment['compound_magnitude'].max()

        # remove any data before the first rep begins, by removing all reps before the first rep that has a max compound magnitude of at least 0.3 * max
        for i in range(len(data_segments)):
            if data_segments[i]['compound_magnitude'].max() >= 0.3 * max:
                data_segments = data_segments[i:]
                break

        # split the data into reps, where each rep is 4 segments
        reps = []
        for i in range(0, len(data_segments), 4):
            reps.append(data_segments[i:i+4])
        # remove the 4th segment of each rep
        for rep in reps:
            rep.pop()

        if plot:
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i in range(len(reps)):
                for j in range(len(reps[i])):
                    plt.plot(reps[i][j]['compound_magnitude'], colours[i])
                    plt.axvspan(reps[i][j].index[0], reps[i]
                                [j].index[-1], facecolor=colours[i], alpha=0.2)
            plt.show()

        return reps

    def createCalibrationData(self, data):
        """
        Creates a calibration dataset by averaging the data in each phase across all reps.

        This can be done on a per exercise type basis, such that later sets can be compared to the calibration set to determine whether the user is performing the exercise correctly.
        """
        setData = self.splitDataIntoReps(self.formatData(data), plot=True)

        calibrations = []

        # remove all reps that dont have length 3
        for rep in setData:
            if len(rep) != 3:
                setData.remove(rep)

        for i in range(0, 3):
            phaseData = []
            for rep in setData:
                phaseData.append(rep[i])

            # Calculate the average time in ms between the last and first data points in each entry in phaseData
            avg_length = int(np.mean(
                [phaseData[i].index[-1] - phaseData[i].index[0] for i in range(len(phaseData))]))

            # Generate a standard time range based on the average length
            standard_time = np.linspace(
                0, avg_length-1, num=avg_length, endpoint=True)

            # use spline interpolation for the data
            scaled_phaseData = []

            for dataset in phaseData:
                dataset_rescaled_index = np.interp(
                    dataset.index, (dataset.index.min(), dataset.index.max()), (0, avg_length-1))
                scaled_data_dict = {}
                for col in dataset.columns:
                    spline = UnivariateSpline(
                        dataset_rescaled_index, dataset[col].values, s=1)
                    scaled_data_dict[col] = spline(standard_time)
                scaled_data = pd.DataFrame(
                    scaled_data_dict, index=standard_time)
                scaled_phaseData.append(scaled_data)

            # Average the scaled data
            avg_data = pd.concat(scaled_phaseData).groupby(level=0).mean()

            if avg_data.isnull().values.any():
                print("NaN values detected after averaging")

            # fit a spline again if needed
            splinefit_data = {}
            for col in avg_data.columns:
                try:
                    # adjust 's' for smoothing
                    spline = UnivariateSpline(
                        avg_data.index, avg_data[col], s=100)
                    splinefit_data[col] = spline
                except Exception as e:
                    print(f"Error during spline fitting for column {col}: {e}")
                    splinefit_data[col] = None

            # Sample the data at avg_length points using the fitted spline
            sampled_index = standard_time
            sampled_data = pd.DataFrame(index=sampled_index)

            for col, spline in splinefit_data.items():
                if spline:
                    sampled_data[col] = spline(sampled_index)

            if sampled_data.isnull().values.any():
                print(f"NaN values detected in sampled data for phase {i+1}")

            calibrations.append(sampled_data)

        self.callibrationData = calibrations

    def visualiseCalibrationSet(self):
        """
        Creates a plot showing the calibration data for each phase.

        Useful for visualising the calibration data and checking for any anomalies.
        """
        # Create a 3x3 grid of subplots
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        # Define the order in which to plot the data for each column
        columns = [['gx', 'gy', 'gz'], ['ax', 'ay', 'az'], ['mx', 'my', 'mz']]
        colors = ['blue', 'red', 'green']

        # Pre-compute the global min and max for each column set
        global_limits = []
        for column_set in columns:
            min_val = float('inf')
            max_val = -float('inf')
            for data in self.callibrationData:
                for column in column_set:
                    min_val = min(min_val, min(data[column]))
                    max_val = max(max_val, max(data[column]))
            global_limits.append((min_val, max_val))

        # Loop through each phase (row) and each sensor (column)
        for i, data in enumerate(self.callibrationData):
            for j, column_set in enumerate(columns):
                for column, color in zip(column_set, colors):
                    axes[i, j].plot(data[column], color=color, label=column)

                # Set the y-axis limits based on the global limits for this column set
                axes[i, j].set_ylim(global_limits[j])

                # Set the title, labels, and legend for each subplot
                title = f"Phase {i+1} - {column_set[0].upper()} series"
                axes[i, j].set_title(title)
                axes[i, j].set_xlabel("Time")
                axes[i, j].set_ylabel("Value")
                axes[i, j].legend()

        plt.tight_layout()
        plt.show()

    def calculateRangeOfMotionScore(self, repData, plot=False):
        """
        Calculates a score for the range of motion of a rep.

        This is calculated by:
        1. Calculating the average of the displacement in phase 1 and 3 of the calibration data
        2. Calculating the displacement in phase 1 and 3 of the rep data
        3. result = (rep displacement / calibration displacement) * 100

        If the result is greater than 100, the rep has exceeded the range of motion of the calibration data, and the excess is subtracted from 100 to give the score.
        e.g. if the rep displacement is 120% of the calibration displacement, the score will be 80.
        """
        calibrationPhase1Displacement = self.calculate_displacement(
            self.callibrationData[0], title="Calibration Phase 1")
        calibrationPhase3Displacement = self.calculate_displacement(
            self.callibrationData[2], title="calibration Phase 3")

        # print(f'Phase 1 displacement for calibration: {calibrationPhase1Displacement}')

        calibrationPhase1DisplacementMagnitude = np.linalg.norm(
            calibrationPhase1Displacement)
        calibrationPhase3DisplacementMagnitude = np.linalg.norm(
            calibrationPhase3Displacement)

        averageCalibrationDisplacement = (
            calibrationPhase1DisplacementMagnitude + calibrationPhase3DisplacementMagnitude) / 2

        repPhase1Displacement = self.calculate_displacement(
            repData[0], title="Rep Phase 1")
        repPhase3Displacement = self.calculate_displacement(
            repData[2], title="Rep Phase 3")

        # print(f'Phase 1 displacement for rep: {repPhase1Displacement}')

        repPhase1DisplacementMagnitude = np.linalg.norm(
            repPhase1Displacement)
        repPhase3DisplacementMagnitude = np.linalg.norm(
            repPhase3Displacement)

        averageRepDisplacement = (
            repPhase1DisplacementMagnitude + repPhase3DisplacementMagnitude) / 2

        # print("Average calibration displacement: " +
        #       str(averageCalibrationDisplacement))
        # print("Average rep displacement: " + str(averageRepDisplacement))

        score = (averageRepDisplacement / averageCalibrationDisplacement) * 100

        return score

    def calculate_displacement(self, data, title="", plot=False):
        """
        Takes a dataframe of data and calculates the displacement.

        The dataframe has to have the following columns:
        gx, gy, gz, ax, ay, az, mx, my, mz

        And an index of time in unix time milliseconds.

        Returns a numpy array of the displacement in each axis.
        """
        # Convert time from milliseconds to seconds for integration purposes
        time_seconds = (data.index - data.index[0]) / 1000

        accelerations = []
        velocities = []
        displacements = []

        # Iterate over each axis to fit a spline, integrate to velocity, then to displacement
        for axis in ['ax', 'ay', 'az']:
            acceleration = data[axis].values
            accelerations.append(acceleration)

            # Integrate the spline to get the velocity
            velocity = cumtrapz(acceleration, time_seconds, initial=0)
            velocities.append(velocity)

            # Integrate the velocity to get the displacement
            displacement = cumtrapz(velocity, time_seconds, initial=0)
            displacements.append(displacement)

        # Combine displacements into a single numpy array
        # Transpose to get shape (n_samples, n_axes)
        displacement_array = np.vstack(displacements).T

        # magnitude is the sum of squares of the sum of displacement

        if plot:
            # 1 x 3 plot
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            # plot the acceleration
            axes[0].plot(time_seconds, accelerations[0], color='blue')
            axes[0].plot(time_seconds, accelerations[1], color='red')
            axes[0].plot(time_seconds, accelerations[2], color='green')
            axes[0].set_title('Acceleration')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Acceleration (m/s^2)')
            axes[0].legend(['x', 'y', 'z'])

            # plot the velocity
            axes[1].plot(time_seconds, velocities[0], color='blue')
            axes[1].plot(time_seconds, velocities[1], color='red')
            axes[1].plot(time_seconds, velocities[2], color='green')
            axes[1].set_title('Velocity')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Velocity (m/s)')
            axes[1].legend(['x', 'y', 'z'])

            # plot the displacement
            axes[2].plot(time_seconds, displacements[0], color='blue')
            axes[2].plot(time_seconds, displacements[1], color='red')
            axes[2].plot(time_seconds, displacements[2], color='green')
            axes[2].set_title('Displacement')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Displacement (m)')
            axes[2].legend(['x', 'y', 'z'])

            plt.tight_layout()
            plt.suptitle(title)
            plt.show()

        maxDisp = np.max(displacement_array, axis=0)
        minDisp = np.min(displacement_array, axis=0)
        return maxDisp - minDisp

    def processSet(self, exerciseData, plot=False):
        setDetails = {
            'Reps': [],
            'ROMScores': [],
            'StabilityScores': [],
            'AverageForces': [],
            'MaxForces': []
        }

        # Process each set of data
        for data in exerciseData:
            setData = self.splitDataIntoReps(self.formatData(data), plot=False)
            setDetails['Reps'].append(len(setData))

            # Range of Motion Scores
            ROMscores = [self.calculateRangeOfMotionScore(
                rep) for rep in setData]
            setDetails['ROMScores'].append(np.mean(ROMscores))

            # Stability Scores
            stabilityScores = []
            for rep in setData:
                stabilityScores.append(
                    np.mean([self.calculateStabilityScore(phase) for phase in rep]))
            setDetails['StabilityScores'].append(np.mean(stabilityScores))

            # Force Calculations
            weight = 10  # Assuming a constant weight for this example
            avgForces, maxForces = [], []
            for rep in setData:
                joinedData = pd.concat(rep)
                avgForces.append(self.averageForce(joinedData.values, weight))
                maxForces.append(self.maxForce(joinedData.values, weight))
            setDetails['AverageForces'].append(np.mean(avgForces))
            setDetails['MaxForces'].append(np.mean(maxForces))

        # Display results for each set
        for i, (reps, ROM, stability, avgF, maxF) in enumerate(zip(setDetails['Reps'], setDetails['ROMScores'],
                                                                   setDetails['StabilityScores'], setDetails['AverageForces'],
                                                                   setDetails['MaxForces'])):
            print(f"Set {i+1}:")
            print(f"  Reps: {reps}")
            print(f"  Range of Motion Score: {ROM:.2f}")
            print(f"  Stability Score: {stability:.2f}")
            print(f"  Average Force: {avgF:.2f} N")
            print(f"  Max Force: {maxF:.2f} N")
            print("\n")

        # Balance Calculations (Comparing sets)
        if len(exerciseData) > 1:
            balanceScores = []
            for i in range(len(exerciseData)-1):
                balanceScore = self.calculateBalanceScore(
                    exerciseData[i], exerciseData[i+1])
                balanceScores.append(balanceScore)
            avgBalanceScore = np.mean(balanceScores)
            print(f"Average Balance Score between sets: {avgBalanceScore:.2f}")
        else:
            print("Only one set available, no balance score calculation.")

    def acceleration_magnitude(self, data):
        return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)

    def averageForce(self, data, weight):
        # Calculate the magnitude of the acceleration vector
        acceleration_magnitude = self.acceleration_magnitude(data)

        # Calculate the force (F = ma)
        force_data = weight * acceleration_magnitude

        # plt.plot(force_data)
        # plt.title("Force Curve")
        # plt.xlabel("Time or Observations")
        # plt.ylabel("Force (N)")
        # plt.show()

        # Calculate and return the average force
        average_force = force_data.mean()
        return average_force

    def maxForce(self, data, weight):
        # Calculate the magnitude of the acceleration vector
        acceleration_magnitude = self.acceleration_magnitude(data)

        # Calculate the force (F = ma)
        force_data = weight * acceleration_magnitude

        # Calculate and return the maximum force
        max_force = force_data.max()
        return max_force

    def scaleData(self, data):
        """
        Scales the given dataframe so that each column is between 0 and 100.
        """
        scaled_data = 100 * (data - data.min()) / (data.max() - data.min())
        return scaled_data

    def calculateBalanceScore(self, df1, df2):
        """
        Takes two dataframes, scales them, and calculates the balance score.

        The balance score is the average of the absolute differences between corresponding
        values in the two datasets, after scaling them between 0 and 100.
        """
        # Ensure that both dataframes have the same columns and length
        if df1.shape != df2.shape or not df1.columns.equals(df2.columns):
            raise ValueError("Dataframes must have the same shape and columns")

        # Scale both dataframes
        scaled_df1 = self.scaleData(df1)
        scaled_df2 = self.scaleData(df2)

        # Calculate the absolute differences
        differences = abs(scaled_df1 - scaled_df2)

        # Calculate the average of these differences
        balance_score = differences.mean().mean()

        return balance_score

    def calculateStabilityScore(self, data, axes=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], plot=False):
        scores = []

        for axis in axes:
            if axis not in data.columns or data[axis].isnull().all():
                # print(f"Warning: Axis {axis} is missing or empty. Skipping.")
                continue

            y_data = data[axis].values
            if len(y_data) == 0:
                # print(f"Warning: No data for axis {axis}. Skipping.")
                continue

            x_data = np.arange(len(y_data))
            try:
                p = Polynomial.fit(x_data, y_data, 2)
                fitted_values = p(x_data)
                mse = mean_squared_error(y_data, fitted_values)
                max_mse = np.var(y_data)
                normalized_mse = mse / max_mse
                score = (1 - normalized_mse) * 100
                scores.append(score)

                if plot:
                    plt.figure(figsize=(10, 4))
                    plt.plot(y_data, label='Actual Data')
                    plt.plot(fitted_values, label='Fitted Curve')
                    plt.title(f'Actual Data vs Fitted Curve - {axis}')
                    plt.xlabel('Time')
                    plt.ylabel(axis)
                    plt.legend()
                    plt.show()

            except ValueError as e:
                pass

        average_score = np.mean(scores) if scores else 0
        return average_score
