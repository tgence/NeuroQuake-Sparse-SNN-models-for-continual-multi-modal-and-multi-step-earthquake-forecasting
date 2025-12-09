import numpy as np
import h5py
import scipy.interpolate as interp
from scipy.signal import find_peaks
import torch
import matplotlib.pyplot as plt
from itertools import product


def resample_data(times, data, target_rate):
    desired_time_step = 1 / target_rate
    resampled_times = np.arange(times[0], times[-1], desired_time_step)
    interpolator = interp.interp1d(times, data, axis=1, kind='linear')
    resampled_data = interpolator(resampled_times)
    return resampled_times, resampled_data

class DataSetStrain:
    def __init__(self):
        self.data_path = './data'
        self.sampling_rate = 50
        self.batch_duration = 4 * 1000
        self.stds = {0: 0.0009698167273583374, 1: 0.0006402534978824718, 2: 0.0004617206345532399, 3: 0.0005666747811344969}
        self.magnitude_ranges = [(min(self.stds[i] for i in range(0,3))*2, 0.03), (0.03, 0.2)]
        self.time_horizons = [(4, 8)]
        self.data = {} # Data
        self.times = {} # Time data
        self.events = {} # Detected events
        self.batches = {} # Data batches
        self.gt_spikes = {}  # Ground truth spike times
        self.number_of_batches = 0
        self.number_of_samples_per_batch = 0

    def fetch_data(self, chunk_id_range):
        for system_id in [1, 2]:
            all_times = []
            all_data = []

            for chunk_id in chunk_id_range:
                with h5py.File(f'{self.data_path}/QS04_023_data{chunk_id}_Elsys{system_id}.mat', 'r') as file:
                    times = file['S/t'][0, :]
                    if system_id == 1:
                        data = file['S/data'][2:]
                    elif system_id == 2:
                        data = file['S/data'][12:]
                    resampled_times, resampled_data = resample_data(times, data, self.sampling_rate)
                    
                    all_times.append(resampled_times)
                    all_data.append(resampled_data)

            # Concatenate all times and data for the system
            self.times[system_id] = np.concatenate(all_times, axis=0)
            self.data[system_id] = np.concatenate(all_data, axis=1)

    def compute_std(self):
        tmp = {channel: [] for channel in range(4)}
        self.fetch_data(range(1, 12)) # Since we have 11 data chunks

        for channel in range(4):
            tmp[channel].append(np.std(self.data[2][channel, :]))


        for channel in range(4):
            self.stds[channel] = np.mean(tmp[channel])
    
        return self.stds




    def split_into_batches(self):
        for system_id, system_data in self.data.items():
            system_times = self.times[system_id]
            batch_duration_sec = self.batch_duration / 1000  # Convert milliseconds to seconds
            n_time_points = len(system_times)
            points_per_batch = int(batch_duration_sec * self.sampling_rate)
            n_batches = n_time_points // points_per_batch  # Number of batches

            # Determine number of sensors based on system_id
            if system_id == 1:
                n_sensors = 7
            elif system_id == 2:
                n_sensors = 4
            else:
                continue  # Skip if system_id is unknown or handle error

            # Initialize the batches tensor
            new_batches = torch.zeros((n_batches, points_per_batch, n_sensors + 1), dtype=torch.float32)  # +1 for time column

            for i in range(n_batches):
                start_idx = i * points_per_batch
                end_idx = start_idx + points_per_batch
                time_batch = system_times[start_idx:end_idx]
                data_batch = system_data[:, start_idx:end_idx]

                new_batches[i, :len(time_batch), 0] = torch.from_numpy(time_batch)  # Time data
                for j in range(data_batch.shape[0]):  # Fill in sensor data
                    new_batches[i, :len(time_batch), j + 1] = torch.from_numpy(data_batch[j])

            self.batches[system_id] = new_batches  # Store batches for each system separately
            self.number_of_batches = n_batches
            self.number_of_samples_per_batch = points_per_batch
            print(f"System {system_id} Batches Shape: {new_batches.shape}")






    def identify_events(self, height_threshold, distance_between_events, batch_num):
        # Channels to consider for system 2
        channels_to_consider = range(0, 4)


        # Initialize or clear events for this batch
        if not hasattr(self, 'events'):
            self.events = {}
        if batch_num not in self.events:
            self.events[batch_num] = {}

        # Initialize event storage for each channel in this batch
        for channel in channels_to_consider:
            self.events[batch_num][channel] = []  # Ensure each channel starts with an empty list of events

        # Process the specified channels of system 2
        for channel in channels_to_consider:
            # Extracting the data and times for the current batch and the next two batches
            combined_signal = np.array(self.batches[2][batch_num][:, channel] )
            combined_times = np.array(self.batches[2][batch_num][:, 0])


            positive_peaks, _ = find_peaks(combined_signal, height=height_threshold)
            negative_peaks, _ = find_peaks(-combined_signal, height=height_threshold)

            # Combine all peaks and sort by their indices (time order)
            all_peaks = np.concatenate((positive_peaks, negative_peaks))
            peak_types = np.concatenate((np.ones_like(positive_peaks), -np.ones_like(negative_peaks)))
            sorted_indices = np.argsort(all_peaks)
            sorted_peaks = all_peaks[sorted_indices]
            sorted_peak_types = peak_types[sorted_indices]
            peak_times = combined_times[sorted_peaks]

            # Examine each peak to form events
            i = 0
            while i < len(sorted_peaks):
                current_peak_index = sorted_peaks[i]
                current_type = sorted_peak_types[i]
                current_peak_time = peak_times[i]
                potential_event_peaks = [(current_peak_time, combined_signal[current_peak_index], current_type)]

                # Look ahead to find other peaks that belong to the same event
                j = i + 1
                while j < len(sorted_peaks) and (peak_times[j] - current_peak_time) <= distance_between_events:
                    potential_event_peaks.append((peak_times[j], combined_signal[sorted_peaks[j]], sorted_peak_types[j]))
                    j += 1

                # Determine the event based on the peaks gathered
                if len(potential_event_peaks) == 1:
                    # Single peak event, treat as its own event
                    self.events[batch_num][channel].append((abs(combined_signal[current_peak_index]), current_peak_time))
                else:
                    # Multiple peaks, find the peak with the largest magnitude
                    max_magnitude = max(abs(p[1]) for p in potential_event_peaks)
                    event_time = potential_event_peaks[0][0]  # Use the time of the first peak as event time
                    self.events[batch_num][channel].append((max_magnitude, event_time))

                i = j  # Move to the next group of peaks


    def test_event_identification(self, CHANNEL, batch_num):
        channel = CHANNEL  # Focusing on PZ2 channel of system 2

        height_threshold = self.stds[channel] * 2  # 3 standard deviations above the mean as the threshold
        distance_between_events = 1 # Minimum Time between events

        # Call identify_events to detect events for system 2
        self.identify_events(height_threshold=height_threshold, distance_between_events=distance_between_events, batch_num=batch_num)
        
        # Access the events for the specified channel in system 2
        events = self.events[batch_num][channel]
        # Extract the magnitudes of the events for the histogram
        event_magnitudes = [event[0] for event in events]
        event_times = [event[1] for event in events]

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

        return event_magnitudes, event_times, height_threshold
    


    def generate_gt_spikes(self, start_time, end_time, batch_num):
        #print(f"Starting GT spike generation for batch {batch_num}...")
        self.gt_spikes[batch_num] = {}

        # Generate spiking patterns only for the relevant channels in system 2
        events_for_batch = self.events[batch_num]
        for channel in events_for_batch:
            self.gt_spikes[batch_num][channel] = {}
            #print(f"Processing channel {channel}...")
            for magnitude_range, time_horizon in product(self.magnitude_ranges, self.time_horizons):
                key = f'Magnitude: {magnitude_range}, Time Horizon: {time_horizon}'
                self.gt_spikes[batch_num][channel][key] = []

                for current_time in np.arange(start_time, end_time, 1.0 / self.sampling_rate):
                    matched_event = any(magnitude_range[0] <= magnitude <= magnitude_range[1] and
                                        time_horizon[0] <= (event_time - current_time) <= time_horizon[1]
                                        for magnitude, event_time in events_for_batch[channel])
                    self.gt_spikes[batch_num][channel][key].append(1 if matched_event else 0)
                

                if len(self.gt_spikes[batch_num][channel][key]) < self.number_of_samples_per_batch:
                    self.gt_spikes[batch_num][channel][key] += [0] * (self.number_of_samples_per_batch - len(self.gt_spikes[batch_num][channel][key]))
                elif len(self.gt_spikes[batch_num][channel][key]) > self.number_of_samples_per_batch:
                    self.gt_spikes[batch_num][channel][key] = self.gt_spikes[batch_num][channel][key][:self.number_of_samples_per_batch]

                #print(f"Finished processing for Magnitude: {magnitude_range}, Time Horizon: {time_horizon}")
            #print(f"Finished processing channel {channel}")
        #print(f"Finished GT spike generation for batch {batch_num}")




    def generate_gt_spikes_for_all_batches(self):
        distance_between_events = 1  # Minimum time between events
        number_of_channels = 4
        number_of_batches = self.batches[2].shape[0]
        print(f"Starting GT spike generation for {number_of_batches} batches ...")

        for batch_num in range(number_of_batches):
            for channel in range(number_of_channels):
                self.identify_events(height_threshold=self.stds[channel] * 2, distance_between_events=distance_between_events, batch_num=batch_num)
            self.generate_gt_spikes(start_time=self.batches[2][batch_num][0, 0], 
                                    end_time=self.batches[2][batch_num][-1, 0], batch_num=batch_num)

        print(f"Finished GT spike generation for {number_of_batches} batches ...")




    def generate_event_labels(self):
        # Number of batches
        number_of_batches = len(self.batches[2])
        
        # Determine the number of channels (assuming all batches have the same structure)
        number_of_channels = 4  # Based on channels_to_consider = range(0, 4)
        
        # Calculate the total number of combinations (C * M * T)
        number_of_combinations = number_of_channels * len(self.magnitude_ranges) * len(self.time_horizons)
        
        # Initialize an empty tensor with the required shape (number_of_batches, 16)
        labels_tensor = torch.zeros((number_of_batches, number_of_combinations))
        
        for batch_num in range(number_of_batches):
            for channel in range(number_of_channels):
                self.identify_events(height_threshold=self.stds[channel] * 2, distance_between_events=1, batch_num=batch_num)
        # Iterate over batches
        print(self.events)
        for batch_num in range(number_of_batches):
            # Call identify_events for the current batch to update self.events
            
            combination_idx = 0
            # Iterate over channels
            for channel in range(number_of_channels):
                # Check if events for the batch and channel exist
                if batch_num in self.events and channel in self.events[batch_num]:
                    # Iterate over magnitude ranges and time horizons
                    for magnitude_range, time_horizon in product(self.magnitude_ranges, self.time_horizons):
                        # Check if there is at least one event that matches the criteria
                        for magnitude, event_time in self.events[batch_num][channel]:
                            if (magnitude_range[0] <= magnitude <= magnitude_range[1] and any(time_horizon[0] <= (event_time - other_event_time) <= time_horizon[1]
                                for _, other_event_time in self.events[batch_num][channel])):
                                    labels_tensor[batch_num, combination_idx] = 1
                                    break
                        combination_idx += 1
                else:
                    # If no events are found, move to the next combination
                    combination_idx += len(self.magnitude_ranges) * len(self.time_horizons)
        
        return labels_tensor


    def generate_classes(self):
        number_of_batches = self.batches[2].shape[0]
        number_of_channels = 4  # Based on channels_to_consider = range(0, 4)
        distance_between_events = 1  # Minimum time between events
        
        # Iterate over batches and identify events
        for batch_num in range(number_of_batches):
            for channel in range(number_of_channels):
                self.identify_events(height_threshold=self.stds[channel] * 2, distance_between_events=distance_between_events, batch_num=batch_num)

        print(self.events)
        # Initialize a list to hold class labels for each batch
        class_labels = []

        # Iterate over batches to determine the class for each batch
        for batch_num in range(number_of_batches - 1):
            next_batch_num = batch_num + 1
            class_label = 1  # Default class

            # Check for events in the next batch
            for channel in range(number_of_channels):
                if next_batch_num in self.events and channel in self.events[next_batch_num]:
                    for magnitude_range_idx, magnitude_range in enumerate(self.magnitude_ranges):
                        for magnitude, event_time in self.events[next_batch_num][channel]:
                            if magnitude_range[0] <= magnitude <= magnitude_range[1]:
                                if magnitude_range_idx == 0:
                                    class_label = max(class_label, 2)  # At least one event in the first magnitude range
                                elif magnitude_range_idx == 1:
                                    if class_label == 2:
                                        class_label = 4  # Events in both magnitude ranges
                                    else:
                                        class_label = max(class_label, 3)  # At least one event in the second magnitude range
            
            class_labels.append(class_label)
        
        # Handle the last batch separately (no next batch to check)
        class_labels.append(1)  # Assuming no events in the next batch for the last batch

        return class_labels



    def convert_gt_spikes_to_tensor(self):
            # Number of batches
            number_of_batches = len(self.gt_spikes)
            
            # Determine the number of channels (assuming all batches have the same structure)
            number_of_channels = len(self.gt_spikes[0])
            
            # Determine the number of magnitude/time horizon combinations in the first batch and first channel
            first_channel_keys = list(self.gt_spikes[0][0].keys())
            number_of_combinations = len(first_channel_keys)
            
            # Determine the number of samples per batch (length of the lists in the first combination)
            number_of_samples_per_batch = len(self.gt_spikes[0][0][first_channel_keys[0]])
            
            # Calculate total number of channels in the final tensor
            total_channels = number_of_channels * number_of_combinations
            
            # Initialize an empty tensor with the required shape (98, 201, 16)
            tensor = torch.zeros((number_of_batches, number_of_samples_per_batch, total_channels))
            
            # Iterate over batches
            for batch_idx in range(number_of_batches):
                # Channel counter for placing values in the correct place in the final tensor
                channel_counter = 0
                # Iterate over channels
                for channel_idx in range(number_of_channels):
                    # Iterate over combinations
                    for combo_idx, key in enumerate(first_channel_keys):
                        value = self.gt_spikes[batch_idx][channel_idx][key]
                        if isinstance(value, list):  # Only process lists
                            tensor[batch_idx, :, channel_counter] = torch.tensor(value)
                        else:
                            raise ValueError(f"Unexpected value type {type(value)} for key {key} in batch {batch_idx}")
                        channel_counter += 1
            
            return tensor









    def test_gt_spikes_generation(self, CHANNEL, batch_num):
        channel = CHANNEL  # Focusing on a specific channel
        height_threshold = self.stds[channel] * 2
        distance_between_events = 1  # Minimum time between events

        # Identify events in the specified channel
        self.identify_events(height_threshold=height_threshold, distance_between_events=distance_between_events, batch_num=batch_num)
        
        # Generate ground truth spiking patterns based on identified events
        self.generate_gt_spikes(start_time=self.batches[2][batch_num][0, 0], 
                                end_time=self.batches[2][batch_num][-1, 0], batch_num=batch_num)

        # Loop through all combinations of magnitude range and time horizon for plotting
        for magnitude_range, time_horizon in product(self.magnitude_ranges, self.time_horizons):
            event_type_key = f'Magnitude: {magnitude_range}, Time Horizon: {time_horizon}'
            spikes = self.gt_spikes[batch_num][channel][event_type_key]
            # generate a list of time stamps of the spikes that are equal to 1 
            spike_times = [i for i in range(len(spikes)) if spikes[i] == 1]
            # Filter actual events to match the chosen magnitude range
            filtered_events = [
                event for event in self.events[batch_num][channel]
                if magnitude_range[0] <= event[0] <= magnitude_range[1]
            ]

            # Extract the times of the filtered events
            filtered_event_times = [event[1] for event in filtered_events]
            indices = [self.sampling_rate*(filtered_event_times[i] - self.batches[2][batch_num][0, 0]) for i in range(len(filtered_event_times))]

            # Plot filtered actual events and GT spiking pattern
            plt.figure(figsize=(10, 4))
            plt.eventplot(indices, lineoffsets=1, linelengths=0.5, color='g', label='Actual Events')
            plt.eventplot(spike_times, lineoffsets=2, linelengths=0.5, color='b', label='GT Spikes')
            plt.title(f'Filtered Actual Events and GT Spiking Pattern for Channel {channel}, {event_type_key}')
            plt.xlabel('Time (s)')
            plt.yticks([1, 2], ['Actual Events', 'GT Spikes'])
            plt.legend()
            plt.show()
