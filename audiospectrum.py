import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
from scipy.fftpack import fft
import time

class AudioStream(object):
    def __init__(self):
        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        self.init_plots()
        self.start_plot()

    def init_plots(self):
        # x variables for plotting
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create line objects for each plot
        self.line_input, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2, label='Input')
        self.line_fft, = ax2.semilogx(xf, np.random.rand(self.CHUNK), '-', lw=2, label='FFT')
        self.line_filtered, = ax3.plot(x, np.random.rand(self.CHUNK), '-', lw=2, label='Filtered')

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        ax1.legend()

        # format spectrum axes
        ax2.set_title('FREQUENCY DOMAIN')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power')
        ax2.set_xlim(20, self.RATE / 2)
        ax2.legend()

        # format filtered waveform axes
        ax3.set_title('LOW-PASS FILTERED WAVEFORM')
        ax3.set_xlabel('Samples')
        ax3.set_ylabel('Volume')
        ax3.set_ylim(0, 255)
        ax3.set_xlim(0, 2 * self.CHUNK)
        ax3.legend()

        # show axes
        plt.tight_layout()
        plt.show(block=False)

    def start_plot(self):
        print('stream started')
        frame_count = 0
        start_time = time.time()

        # Moving average filter parameters
        filter_window_size = 20
        filter_window = np.ones(filter_window_size) / filter_window_size

        while not self.pause:
            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            self.data_np = np.array(data_int, dtype='b')[::2] + 128

            # Compute FFT
            yf = fft(data_int)
            self.data_fft = np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK)

            # Apply low-pass filter
            self.filtered_data_np = np.convolve(self.data_np, filter_window, mode='same')

            # Update plots
            self.line_input.set_ydata(self.data_np)
            self.line_fft.set_ydata(self.data_fft)
            self.line_filtered.set_ydata(self.filtered_data_np)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)

    def onClick(self, event):
        self.pause = True

if __name__ == '__main__':
    AudioStream()
