"""Test-only oracle: the per-window computation of `data/csi_proc_python/processing.py`.

Source: the method loop (`for method in output`) and the metrics-window feature block of that
script, copied statement by statement for one window of `csi` (shape `(antennas, packets,
subcarriers)`, antennas 0 and 1, `scale = 1`, `norm = 0`, `n_weights = 40`, `slide = 1`).
Two deliberate deviations: the metrics window covers all `len(theta_star_out2)` inner windows
instead of sliding over a fixed 10, and `fix_distance_loop` swaps the script's pair loop for
the unique-pair loop (`next > sample`).
"""

import numpy as np

n_weights = 40
deltaTheta = 2 * np.pi / n_weights
slide = 1


def paper_features(csi: np.ndarray, fix_distance_loop: bool = False) -> list[float]:
    csi_ch1 = csi[0]
    csi_ch2 = csi[1]
    n_samples = csi_ch1.shape[0]

    theta_star_out1, theta_star_out2, theta_star_out3 = [], [], []
    max_int, max_distance, min_variance = [], [], []

    for method in [1, 2, 3]:
        if method == 1:
            window_size = 8
        if method == 2:
            window_size = 16
        if method == 3:
            window_size = 1
        start_id = 1
        stop_id = start_id + window_size

        while stop_id <= n_samples:
            window_idx = np.arange(start_id, stop_id)
            start_id = start_id + slide
            stop_id = start_id + window_size

            spectrum_integral = np.zeros((n_weights, 2))
            distance_arr = np.zeros((n_weights, 2))
            mean_var = np.zeros((n_weights, 2))

            w_csi_ch1 = csi_ch1[window_idx, :]
            w_csi_ch2 = csi_ch2[window_idx, :]

            comb_count = 0
            theta1 = 0

            for _ in np.arange(0, n_weights):
                theta1 = theta1 + deltaTheta
                csi_comb = w_csi_ch1 + w_csi_ch2 * np.exp(theta1 * 1j)
                amp_csi_comb = abs(csi_comb)

                theta_save = theta1
                if theta_save > np.pi:
                    theta_save = theta_save - (2 * np.pi)

                DopplerSpec_comb = np.mean(abs(np.fft.fft(amp_csi_comb, axis=0)), axis=1)

                if method == 1:
                    f_int = sum(DopplerSpec_comb)
                    spectrum_integral[comb_count, :] = np.array([f_int, theta_save])
                    comb_count = comb_count + 1

                if method == 2:
                    distance = []
                    for sample in range(csi_comb.shape[0] - 1):
                        vector1 = amp_csi_comb[sample, :]
                        first = sample + 1 if fix_distance_loop else 1
                        for next in range(first, csi_comb.shape[0]):
                            vector2 = amp_csi_comb[next, :]
                            distance.append(np.sqrt(np.sum((vector2 - vector1) ** 2)))
                    distance_arr[comb_count, :] = np.array([np.mean(distance), theta_save])
                    comb_count = comb_count + 1

                if method == 3:
                    variance = np.var(amp_csi_comb, axis=1)
                    mean_variance = np.mean(variance)
                    mean_var[comb_count, :] = np.array([mean_variance, theta_save])
                    comb_count = comb_count + 1

            if method == 1:
                theta_star_index = np.argmax(spectrum_integral[:, 0])
                theta_star_out1.append(spectrum_integral[theta_star_index, 1])
                max_int.append(np.max(spectrum_integral[:, 0]))
            if method == 2:
                theta_star_index = np.argmax(distance_arr[:, 0])
                theta_star_out2.append(distance_arr[theta_star_index, 1])
                max_distance.append(np.max(distance_arr[:, 0]))
            if method == 3:
                theta_star_index = np.argmin(mean_var[:, 0])
                theta_star_out3.append(mean_var[theta_star_index, 1])
                min_variance.append(np.min(mean_var[:, 0]))

    theta_star_out1 = np.array(theta_star_out1)
    theta_star_out2 = np.array(theta_star_out2)
    theta_star_out3 = np.array(theta_star_out3)
    max_int = np.array(max_int)
    max_distance = np.array(max_distance)
    min_variance = np.array(min_variance)

    window_idx_2 = np.arange(0, len(theta_star_out2))
    return [
        float(np.var(theta_star_out1[window_idx_2])),
        float(np.mean(max_int[window_idx_2])),
        float(np.var(theta_star_out2[window_idx_2])),
        float(np.mean(max_distance[window_idx_2])),
        float(np.var(theta_star_out3[window_idx_2])),
        float(np.mean(min_variance[window_idx_2])),
    ]
