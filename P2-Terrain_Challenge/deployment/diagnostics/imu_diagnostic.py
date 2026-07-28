#!/usr/bin/env python3
"""IMU diagnostic and calibration tool for Mini Pupper 2.

Usage:
  python imu_diagnostic.py              # 10s capture, static + walking
  python imu_diagnostic.py --duration 20 # longer capture
  python imu_diagnostic.py --live        # live streaming mode

Measures:
  - Gyro bias and noise floor (static)
  - Accelerometer gravity alignment
  - Walking vibration spectrum
  - Velocity estimation quality (if walking)
"""

import numpy as np
import time
import argparse
from collections import deque

from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class IMUDiagnostic:
    def __init__(self):
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.esp32 = self.hardware.pwm_params.esp32
        time.sleep(0.3)

        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81

    def read_imu(self):
        """Read IMU and return (gyro_rad_s, accel_m_s2)."""
        data = self.esp32.imu_get_data()
        gyro_raw = np.array([data[0], data[1], data[2]], dtype=np.float64)
        accel_raw = np.array([data[3], data[4], data[5]], dtype=np.float64)
        return gyro_raw * self.gyro_scale, accel_raw * self.accel_scale

    def capture(self, duration=10.0, rate=50):
        """Capture IMU data at given rate for given duration."""
        dt = 1.0 / rate
        n_samples = int(duration * rate)
        gyro_data = np.zeros((n_samples, 3))
        accel_data = np.zeros((n_samples, 3))
        timestamps = np.zeros(n_samples)

        print(f"[IMU] Capturing {duration}s at {rate}Hz ({n_samples} samples)...")
        t0 = time.time()

        for i in range(n_samples):
            step_start = time.time()
            gyro, accel = self.read_imu()
            gyro_data[i] = gyro
            accel_data[i] = accel
            timestamps[i] = time.time() - t0

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if (i + 1) % rate == 0:
                print(f"  {i+1}/{n_samples} samples...")

        actual_rate = n_samples / (time.time() - t0)
        print(f"[IMU] Captured {n_samples} samples (actual rate: {actual_rate:.1f}Hz)")
        return timestamps, gyro_data, accel_data

    def analyze_static(self, gyro_data, accel_data):
        """Analyze static (stationary) IMU data."""
        print("\n" + "=" * 60)
        print("STATIC IMU ANALYSIS")
        print("=" * 60)

        # Gyro bias
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        print(f"\nGyroscope (rad/s):")
        print(f"  Bias (mean):  X={gyro_mean[0]:+.4f}  Y={gyro_mean[1]:+.4f}  Z={gyro_mean[2]:+.4f}")
        print(f"  Noise (std):  X={gyro_std[0]:.4f}  Y={gyro_std[1]:.4f}  Z={gyro_std[2]:.4f}")
        print(f"  Noise (deg/s): X={gyro_std[0]*180/np.pi:.2f}  Y={gyro_std[1]*180/np.pi:.2f}  Z={gyro_std[2]*180/np.pi:.2f}")

        # Accel gravity
        accel_mean = np.mean(accel_data, axis=0)
        accel_std = np.std(accel_data, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        gravity_dir = accel_mean / accel_norm if accel_norm > 0.1 else np.array([0, 0, -1])

        print(f"\nAccelerometer (m/s²):")
        print(f"  Mean:     X={accel_mean[0]:+.3f}  Y={accel_mean[1]:+.3f}  Z={accel_mean[2]:+.3f}")
        print(f"  Noise:    X={accel_std[0]:.3f}  Y={accel_std[1]:.3f}  Z={accel_std[2]:.3f}")
        print(f"  |gravity|: {accel_norm:.3f} m/s² (expected: 9.81)")
        print(f"  Gravity direction: [{gravity_dir[0]:+.4f}, {gravity_dir[1]:+.4f}, {gravity_dir[2]:+.4f}]")
        print(f"  Tilt from vertical: {np.arccos(abs(gravity_dir[2]))*180/np.pi:.1f}°")

        # Velocity drift estimate
        accel_detrended = accel_data - accel_mean  # remove gravity
        drift_1s = np.sqrt(np.mean(accel_std**2)) * 1.0  # RMS drift per second
        print(f"\nVelocity estimation quality:")
        print(f"  Gravity-subtracted accel noise: {np.mean(accel_std):.3f} m/s² (RMS)")
        print(f"  Expected velocity drift: {drift_1s:.3f} m/s per second")
        print(f"  Walking speed signal: ~0.15 m/s")
        print(f"  SNR: {0.15/drift_1s:.2f} ({'USABLE' if 0.15/drift_1s > 1.0 else 'NOT usable for velocity'})")

        return gyro_mean, gravity_dir

    def analyze_walking(self, gyro_data, accel_data, timestamps, rate=50):
        """Analyze walking IMU data — vibration spectrum and velocity estimate."""
        print("\n" + "=" * 60)
        print("WALKING IMU ANALYSIS")
        print("=" * 60)

        # Vibration spectrum
        from numpy.fft import fft, fftfreq
        n = len(accel_data)
        freqs = fftfreq(n, 1.0/rate)[:n//2]

        for axis, name in enumerate(['X (forward)', 'Y (lateral)', 'Z (vertical)']):
            spectrum = np.abs(fft(accel_data[:, axis]))[:n//2]
            spectrum[0] = 0  # remove DC
            peak_idx = np.argmax(spectrum)
            peak_freq = freqs[peak_idx]
            print(f"\n  Accel {name}:")
            print(f"    Peak frequency: {peak_freq:.1f} Hz")
            print(f"    RMS amplitude:  {np.std(accel_data[:, axis]):.2f} m/s²")

            # Power in bands
            for low, high, label in [(0, 2, '0-2Hz (locomotion)'),
                                      (2, 5, '2-5Hz (gait)'),
                                      (5, 12, '5-12Hz (vibration)')]:
                mask = (freqs >= low) & (freqs < high)
                band_power = np.sum(spectrum[mask]**2)
                total_power = np.sum(spectrum**2)
                pct = 100 * band_power / total_power if total_power > 0 else 0
                print(f"    Power {label}: {pct:.1f}%")

        # Naive velocity integration (to show drift)
        dt = 1.0 / rate
        gravity_est = np.mean(accel_data[:min(50, n)], axis=0)  # first 1s as gravity ref
        lin_accel = accel_data - gravity_est
        velocity = np.cumsum(lin_accel * dt, axis=0)

        print(f"\n  Naive velocity integration (no filter):")
        print(f"    Final velocity: X={velocity[-1,0]:+.2f}  Y={velocity[-1,1]:+.2f}  Z={velocity[-1,2]:+.2f} m/s")
        print(f"    Max velocity:   {np.max(np.abs(velocity)):.2f} m/s")
        print(f"    Expected:       ~0.15 m/s forward")
        print(f"    Drift ratio:    {np.max(np.abs(velocity))/0.15:.1f}x target")

        # Complementary filter velocity
        alpha_v = 0.95
        vel_cf = np.zeros(3)
        vel_cf_history = np.zeros((n, 3))
        for i in range(n):
            lin_acc = accel_data[i] - gravity_est
            vel_cf = alpha_v * (vel_cf + lin_acc * dt)
            vel_cf_history[i] = vel_cf

        print(f"\n  Complementary filter velocity (α={alpha_v}):")
        print(f"    Mean: X={np.mean(vel_cf_history[:,0]):+.3f}  Y={np.mean(vel_cf_history[:,1]):+.3f}  Z={np.mean(vel_cf_history[:,2]):+.3f}")
        print(f"    Std:  X={np.std(vel_cf_history[:,0]):.3f}  Y={np.std(vel_cf_history[:,1]):.3f}  Z={np.std(vel_cf_history[:,2]):.3f}")
        print(f"    Range X: [{np.min(vel_cf_history[:,0]):+.3f}, {np.max(vel_cf_history[:,0]):+.3f}]")

        # Angular velocity during walking
        print(f"\n  Gyroscope during walking:")
        print(f"    Mean: X={np.mean(gyro_data[:,0]):+.3f}  Y={np.mean(gyro_data[:,1]):+.3f}  Z={np.mean(gyro_data[:,2]):+.3f} rad/s")
        print(f"    Std:  X={np.std(gyro_data[:,0]):.3f}  Y={np.std(gyro_data[:,1]):.3f}  Z={np.std(gyro_data[:,2]):.3f} rad/s")
        print(f"    Range: [{np.min(gyro_data):.2f}, {np.max(gyro_data):.2f}] rad/s")

    def live_stream(self, rate=25):
        """Stream IMU readings to terminal."""
        dt = 1.0 / rate
        # Calibrate for 2s
        print("[IMU] Calibrating (hold still for 2s)...")
        cal_gyro, cal_accel = [], []
        for _ in range(int(2 * rate)):
            g, a = self.read_imu()
            cal_gyro.append(g)
            cal_accel.append(a)
            time.sleep(dt)
        gyro_bias = np.mean(cal_gyro, axis=0)
        gravity = np.mean(cal_accel, axis=0)
        print(f"[IMU] Gyro bias: {gyro_bias}")
        print(f"[IMU] Gravity: {gravity} ({np.linalg.norm(gravity):.2f} m/s²)")

        # Complementary filter state
        alpha_v = 0.95
        vel = np.zeros(3)
        grav_est = gravity / np.linalg.norm(gravity)

        print("\n[IMU] Streaming (Ctrl+C to stop)...")
        print(f"{'ang_vel_x':>10} {'ang_vel_y':>10} {'ang_vel_z':>10} | {'vel_x':>8} {'vel_y':>8} {'vel_z':>8} | {'grav_x':>7} {'grav_y':>7} {'grav_z':>7}")
        print("-" * 95)

        try:
            while True:
                step_start = time.time()
                gyro, accel = self.read_imu()

                omega = gyro - gyro_bias
                # Update gravity estimate
                g_gyro = grav_est - np.cross(omega, grav_est) * dt
                a_norm = np.linalg.norm(accel)
                if a_norm > 0.1:
                    g_accel = accel / a_norm
                else:
                    g_accel = grav_est
                grav_est = 0.98 * g_gyro + 0.02 * g_accel
                gn = np.linalg.norm(grav_est)
                if gn > 0.1:
                    grav_est /= gn

                lin_accel = accel - grav_est * 9.81
                vel = alpha_v * (vel + lin_accel * dt)

                print(f"{omega[0]:>+10.3f} {omega[1]:>+10.3f} {omega[2]:>+10.3f} | "
                      f"{vel[0]:>+8.3f} {vel[1]:>+8.3f} {vel[2]:>+8.3f} | "
                      f"{grav_est[0]:>+7.3f} {grav_est[1]:>+7.3f} {grav_est[2]:>+7.3f}",
                      end='\r')

                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\n[IMU] Stopped.")


def main():
    parser = argparse.ArgumentParser(description='IMU Diagnostic Tool')
    parser.add_argument('--duration', type=float, default=10.0, help='Capture duration (seconds)')
    parser.add_argument('--rate', type=int, default=50, help='Sample rate (Hz)')
    parser.add_argument('--live', action='store_true', help='Live streaming mode')
    parser.add_argument('--walking', action='store_true', help='Analyze walking data (vs static)')
    args = parser.parse_args()

    diag = IMUDiagnostic()

    if args.live:
        diag.live_stream(rate=args.rate)
        return

    if not args.walking:
        print("\n[STATIC TEST] Keep robot still and level...")
        time.sleep(2)
        ts, gyro, accel = diag.capture(duration=args.duration, rate=args.rate)
        gyro_bias, gravity_dir = diag.analyze_static(gyro, accel)
        print(f"\n[CALIBRATION VALUES]")
        print(f"  gyro_offset = np.array([{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}])")
        print(f"  gravity_ref = np.array([{gravity_dir[0]:.4f}, {gravity_dir[1]:.4f}, {gravity_dir[2]:.4f}])")
    else:
        print("\n[WALKING TEST] Start walking the robot, then press Enter...")
        input()
        ts, gyro, accel = diag.capture(duration=args.duration, rate=args.rate)
        diag.analyze_walking(gyro, accel, ts, rate=args.rate)

    # Save raw data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = 'walking' if args.walking else 'static'
    fname = f"/home/ubuntu/mp2_mlp/imu_{mode}_{timestamp}.npz"
    np.savez(fname, timestamps=ts, gyro=gyro, accel=accel)
    print(f"\n[SAVED] Raw data: {fname}")


if __name__ == '__main__':
    main()
