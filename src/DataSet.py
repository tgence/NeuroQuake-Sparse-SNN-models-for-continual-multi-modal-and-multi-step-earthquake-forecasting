import numpy as np
import h5py
import scipy.interpolate as interp
from scipy.signal import find_peaks
from itertools import product
from scipy.signal import correlate
from scipy.stats import median_abs_deviation


def resample_data(times, data, target_rate):
    # Resample the data to a target sampling rate using linear interpolation
    desired_time_step = 1 / target_rate
    resampled_times = np.arange(times[0], times[-1], desired_time_step)
    interpolator = interp.interp1d(times, data, axis=1, kind='linear')
    resampled_data = interpolator(resampled_times)
    return resampled_times, resampled_data


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def find_threshold(data, mad_multiplier):
    # Calculate the median of the data
    median = np.median(data)

    # Calculate the Median Absolute Deviation of the data
    mad = median_abs_deviation(data)

    # Calculate the threshold
    threshold_p = median + mad_multiplier * mad
    threshold_n = median - mad_multiplier * mad

    return threshold_p, threshold_n

class DataSet:
    def __init__(self, params):
        self.data_path = params['data_path']
        self.sampling_rate = params['sampling_rate']
        self.magnitude_ranges = params['magnitude_ranges']
        self.time_horizons = params['time_horizons']
        self.data = {}  # Dictionary to hold data for both systems
        self.times = {}  # Dictionary to hold times for both systems
        self.batches = {}  # List to hold data batches
        self.events = {}  # Dictionary to hold detected events for each PZ channel
        self.gt_spike_times = {}  # Dictionary to hold ground truth spike times
        self.encoded_spikes = {}
        self.current_batch = 0

    def fetch_data(self, chunk_id):
        # Load data for both system 1 and 2
        for system_id in [1, 2]:
            with h5py.File(f'{self.data_path}/QS04_023_data{chunk_id}_Elsys{system_id}.mat', 'r') as f:
                times = f['S/t'][0, :]
                data = f['S/data']
                resampled_times, resampled_data = resample_data(times, data, self.sampling_rate)
                self.times[system_id] = resampled_times
                self.data[system_id] = resampled_data

    def split_into_batches(self, batch_duration):

        # Iterate over each system to split data into batches
        for system_id, system_data in self.data.items():
            system_times = self.times[system_id]
            batch_duration_sec = batch_duration / 1000  # Convert milliseconds to seconds
            total_chunk_duration = system_times[-1] - system_times[0]
            n_time_points = len(system_times)
            n_batches = total_chunk_duration / batch_duration_sec
            points_per_batch = int(n_time_points / n_batches)  # Data points per batch

            # Initialize an empty list for the batches of the current system
            self.batches[system_id] = []

            for start_idx in range(0, n_time_points, points_per_batch):
                end_idx = min(start_idx + points_per_batch, n_time_points)  # Ensure we don't go beyond the array
                time_batch = system_times[start_idx:end_idx]
                data_batch = system_data[:, start_idx:end_idx]
                # Append the time and data batches to the list for the current system
                self.batches[system_id].append((time_batch, data_batch))

    def identify_events(self, height_threshold, distance_between_events):
        # Channels to consider for system 2
        channels_to_consider = range(12, 16)

        # Process the specified channels of system 2
        for channel in channels_to_consider:
            signal = self.data[2][channel, :]  # Assuming self.data[2] holds data for system 2
            peaks, _ = find_peaks(signal, height=height_threshold, distance=distance_between_events)
            negative_peaks, _ = find_peaks(-signal, height=height_threshold, distance=distance_between_events)

            magnitudes = signal[peaks]
            negative_magnitudes = -signal[negative_peaks]

            all_peaks = np.concatenate((peaks, negative_peaks))
            all_magnitudes = np.concatenate((magnitudes, negative_magnitudes))
            sorted_indices = np.argsort(all_peaks)
            all_peaks_sorted = all_peaks[sorted_indices]
            all_magnitudes_sorted = all_magnitudes[sorted_indices]

            event_times = self.times[2][all_peaks_sorted]  # Assuming self.times[2] holds times for system 2
            events_for_channel = [(magnitude, time) for magnitude, time in zip(all_magnitudes_sorted, event_times)]

            # Store the identified events for each channel in system 2
            self.events[channel] = events_for_channel

    def generate_gt_spiking_patterns(self, start_time, end_time):
        # Generate spiking patterns only for the relevant channels in system 2
        for channel in self.events:
            self.gt_spike_times[channel] = {}

            for magnitude_range, time_horizon in product(self.magnitude_ranges, self.time_horizons):
                key = f'Magnitude: {magnitude_range}, Time Horizon: {time_horizon}'
                self.gt_spike_times[channel][key] = []

                for current_time in np.arange(start_time, end_time, 1.0 / self.sampling_rate):
                    for magnitude, event_time in self.events[channel]:
                        if (magnitude_range[0] <= magnitude <= magnitude_range[1] and
                                time_horizon[0] <= (event_time - current_time) <= time_horizon[1]):
                            self.gt_spike_times[channel][key].append(current_time)
                            break  # Found an event that meets the criteria

    def delta_spike_encode(self, mad_multipliers, batch_num):
        # Get the specified batch for each system
        for system_id, batches in self.batches.items():
            print(f"Processing system {system_id}")

            # Check if the batch number is valid
            if batch_num >= len(batches):
                print(f"Invalid batch number {batch_num} for system {system_id}")
                continue

            # Get the specified batch
            system_data = batches[batch_num]

            print(f"Processing batch {batch_num} in system {system_id}")

            # Initialize an empty list to hold the encoded spikes for the current system
            self.encoded_spikes[system_id] = []

            # Iterate over each channel in the current system
            for channel_idx, channel_data in enumerate(
                    system_data[1]):  # system_data[1] because system_data is a tuple (time_batch, data_batch)
                print(f"Processing channel {channel_idx} in batch {batch_num} in system {system_id}")
                # Initialize an empty list to hold the encoded spikes for the current channel
                channel_spikes = []

                # Calculate the window size and threshold for the current channel
                window_size = 5000
                # Apply moving average filter
                filtered_data = moving_average(channel_data, window_size)
                # Calculate the derivative of the filtered data
                derivative = np.diff(filtered_data)

                # Get the mad_multiplier for the current channel
                mad_multiplier = mad_multipliers[system_id][channel_idx]

                threshold_p, threshold_n = find_threshold(derivative, mad_multiplier)

                # Find indices where the absolute difference exceeds the positive or negative threshold
                spike_indices_p = np.where(derivative > threshold_p)[0]
                spike_indices_n = np.where(derivative < threshold_n)[0]

                # Store the time of the spikes instead of just a binary value
                # Use the middle of the window as the spike time
                spike_times_p = system_data[0][spike_indices_p + window_size // 2]
                spike_times_n = system_data[0][spike_indices_n + window_size // 2]

                # Append spike times to channel_spikes
                channel_spikes.extend(spike_times_p)
                channel_spikes.extend(spike_times_n)
                # Add the encoded spikes for the current channel to the list for the current system
                self.encoded_spikes[system_id].append(channel_spikes)
                print(
                    f"Finished processing channel {channel_idx} in batch {batch_num} in system {system_id}. Encoded {len(channel_spikes)} spikes.")

            print(f"Finished processing batch {batch_num} in system {system_id}")

    def get_next_batch(self):
        """Retrieves the next batch of data."""
        if self.current_batch < len(self.batches):
            batch = self.batches[self.current_batch]
            self.current_batch += 1
            return batch
        else:
            return None  # No more batches
