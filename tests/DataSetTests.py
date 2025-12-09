import unittest
import matplotlib.pyplot as plt
from src.DataSet import *
import numpy as np


class DatasetTests(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.chunk_id = 4
        self.batch_duration = 80000  # Duration of each batch in milliseconds
        self.params = {
            'data_path': '../data',
            'sampling_rate': 5000,
            'magnitude_ranges': [(0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05)],
            'time_horizons': [(0, 1), (1, 4), (4, 8), (8, 16)]
        }
        self.mad_multipliers = {
            1: [6, 6, 6, 6, 6, 6, 6, 6, 6],  # System 1 has 13 channels
            2: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12]  # System 2 has 16 channels
        }
        self.dataset = DataSet(self.params)
        self.dataset.fetch_data(chunk_id=self.chunk_id)  # Fetch the chunk
        self.batches = self.dataset.split_into_batches(self.batch_duration)

    def test_gt_spiking_patterns_generation(self):
        channel = 13  # Focusing on one of the last four channels of system 2
        height_threshold = np.std(
            self.dataset.data[2][channel, :]) * 3  # 3 standard deviations above the mean as the threshold
        distance_between_events = 100  # Minimum sample distance between events

        # Identify events in the specified channel
        self.dataset.identify_events(height_threshold=height_threshold, distance_between_events=distance_between_events)

        # Generate ground truth spiking patterns based on identified events
        self.dataset.generate_gt_spiking_patterns(start_time=0, end_time=self.dataset.times[2][-1])

        # Choose the first combination of magnitude range and time horizon for plotting
        magnitude_range, time_horizon = self.params['magnitude_ranges'][0], self.params['time_horizons'][0]
        event_type_key = f'Magnitude: {magnitude_range}, Time Horizon: {time_horizon}'
        spike_times = self.dataset.gt_spike_times[channel][event_type_key]

        # Filter actual events to match the chosen magnitude range
        filtered_events = [
            event for event in self.dataset.events[channel]
            if magnitude_range[0] <= event[0] <= magnitude_range[1]
        ]

        # Extract the times of the filtered events
        filtered_event_times = [event[1] for event in filtered_events]

        # Plot filtered actual events
        plt.eventplot(filtered_event_times, lineoffsets=1, linelengths=0.5, color='g', label='Actual Events')

        # Plot GT spiking pattern
        plt.eventplot(spike_times, lineoffsets=2, linelengths=0.5, color='b', label='GT Spikes')

        # Add some visual aids and labels to the plot
        plt.title(f'Filtered Actual Events and GT Spiking Pattern for Channel {channel}, {event_type_key}')
        plt.xlabel('Time (s)')
        plt.yticks([1, 2], ['Actual Events', 'GT Spikes'])
        plt.legend()

        plt.show()

    def test_creation_of_data_batches(self):

        fig, ax = plt.subplots(figsize=(16, 8))

        # Directly access batches for each system from the dictionary
        batches1 = self.dataset.batches[1] if 1 in self.dataset.batches else []
        batches2 = self.dataset.batches[2] if 2 in self.dataset.batches else []

        ax2 = ax3 = None  # Initialize secondary axes to None

        if batches1:  # Plot data for System 1 if available
            # Example: Plotting data from a specific channel in System 1
            cf_s = 0.374  # Conversion factor for System 1
            time_batch, data_batch = batches1[1]  # First batch of System 1
            ax.plot(time_batch, data_batch[2, :] * cf_s * 1e3, c="tab:blue", label="S1 (Strain)")
            ax.set_ylabel(r"$\delta \; [mm]$")
            ax.set_xlabel("Time (s)")
            ax.tick_params(axis='y', colors='tab:blue')

        if batches2:  # Plot data for System 2 if available
            # Example: Plotting Eddy current sensor data from System 2
            cf_e = 0.000128  # Conversion factor for Eddy current sensors in System 2
            time_batch, data_batch = batches2[1]  # First batch of System 2
            ax2 = ax.twinx()  # Create a secondary y-axis for System 2 data
            ax2.tick_params(axis='y', colors='tab:orange')
            ax2.plot(time_batch, -data_batch[3, :] * cf_e * 1e3, c="tab:orange", label="E1 (Slip)")
            ax2.set_ylabel(r"$\Delta \delta \; [mm]$")

            # Example: Plotting Piezoelectric sensor data from System 2
            ax3 = ax.twinx()  # Create a third y-axis for additional System 2 data
            ax3.tick_params(axis='y', colors='tab:green')
            ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
            ax3.plot(time_batch, data_batch[12, :], c="tab:green", label="PZ1 (AEs)")

        # Only call legend() if the corresponding axes have been defined
        ax.legend(loc="upper center") if batches1 else None
        ax2.legend(loc="upper right") if ax2 else None
        ax3.legend(loc="lower right") if ax3 else None

        plt.show()

    def test_event_identification(self):
        channel = 13  # Focusing on one of the last four channels of system 2
        height_threshold = np.std(
            self.dataset.data[2][channel, :]) * 3  # 3 standard deviations above the mean as the threshold
        distance_between_events = 100  # Minimum sample distance between events

        # Call identify_events to detect events for system 2
        self.dataset.identify_events(height_threshold=height_threshold, distance_between_events=distance_between_events)

        # Access the events for the specified channel in system 2
        events = self.dataset.events[channel]

        # Extract the magnitudes of the events for the histogram
        event_magnitudes = [event[0] for event in events]

        # Define the bins for the histogram
        bin_width = 0.005  # Width of each bin
        min_magnitude = min(event_magnitudes)
        max_magnitude = max(event_magnitudes)
        bins = np.arange(min_magnitude, max_magnitude + bin_width, bin_width)

        # Plot histogram of event magnitudes
        plt.figure(figsize=(10, 6))
        plt.hist(event_magnitudes, bins=bins, color='skyblue', edgecolor='black')
        plt.title('Histogram of Event Magnitudes')
        plt.xlabel('Magnitude')
        plt.ylabel('Counts')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    def test_delta_spike_encode(self):
        # Perform delta spike encoding for a single batch
        batch_num = 1  # Example value, adjust as needed
        self.dataset.delta_spike_encode(self.mad_multipliers, batch_num)

        # Check if encoded_spikes is empty
        if not self.dataset.encoded_spikes:
            print("No spikes were encoded.")
            return

        # Initialize a figure for the plot
        plt.figure(figsize=(10, 6))
        # Initialize a counter for the lineoffsets
        lineoffset_counter = 0
        # Define the labels for the sensors
        sensor_labels = ['S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'PZ1', 'PZ2', 'PZ3', 'PZ4']
        # Plot spike events for the selected channels of system 1
        for channel_idx in range(2, 6):
            plt.eventplot(self.dataset.encoded_spikes[1][channel_idx], lineoffsets=lineoffset_counter,
                          linelengths=0.5, color='b')
            lineoffset_counter += 1  # Increment the counter
        # Plot spike events for the selected channels of system 2
        for channel_idx in range(0, 4):
            plt.eventplot(self.dataset.encoded_spikes[2][channel_idx], lineoffsets=lineoffset_counter,
                          linelengths=0.5, color='r')
            lineoffset_counter += 1  # Increment the counter
        # Plot spike events for the last 4 channels of system 2
        for channel_idx in range(12, 16):
            plt.eventplot(self.dataset.encoded_spikes[2][channel_idx], lineoffsets=lineoffset_counter,
                          linelengths=0.5, color='g')
            lineoffset_counter += 1  # Increment the counter

        # Add labels and title to the plot
        plt.title('Spike Patterns for Different Channels')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel Index')
        # Set the limits of the y-axis
        plt.ylim(-1, 12)  # Adjust the y-axis limit to match the number of channels

        times = self.dataset.batches[1][batch_num][0]
        plt.xlim(times[0], times[-1])  # Adjust the x-axis limit to match the time range of the batch
        # Set the y-ticks and their labels
        plt.yticks(range(12), sensor_labels)
        # Show the plot
        plt.show()

    def test_signal_feature_encoding(self):
        batch_num = 1  # Example value, adjust as needed
        multiplier = 6  # Example value, adjust as needed
        # Select the strain gauge channel
        system_id = 1  # System
        channel_idx = 2

        # Get the time and data for the selected channel from the specified batch
        time_batch = self.dataset.batches[system_id][batch_num][0]
        channel_data = self.dataset.batches[system_id][batch_num][1][channel_idx, :]

        # Apply the moving average filter
        window_size = 5000
        filtered_data = moving_average(channel_data, window_size)

        # Calculate the derivative of the filtered data
        derivative = np.diff(filtered_data)

        # Calculate the threshold for the batch data
        threshold_p, threshold_n = find_threshold(derivative, multiplier)

        # Adjust the time array to match the length of the derivative array
        time_batch_adjusted = time_batch[:len(derivative)]

        # Plot the derivative of the filtered data
        plt.figure(figsize=(10, 6))
        plt.plot(time_batch_adjusted, derivative)
        plt.axhline(y=threshold_p, color='r', linestyle='--')  # Plot the threshold as a horizontal line
        plt.axhline(y=threshold_n, color='r', linestyle='--')  # Plot the threshold as a horizontal line
        plt.title('Derivative of Filtered Signal and Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Derivative')
        plt.show()


if __name__ == '__main__':
    unittest.main()
