import multiprocessing as mp
import queue
from typing import List, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from schema import Schema, Optional

from config import Param, Config

Figure = Any  # Stub for matplotlib

# TODO: Definir mejores defaults
DEFAULT_ANIMATION_INTERVAL: int = 1000  # Data is processed every 100 ms
DEFAULT_RENDER_STEP: int = 10  # Plot is re-rendered every ANIMATION_INTERVAL*render_step ms


class AsyncPlotter:

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.end_event = mp.Event()
        self.new_w_provider: mp.Queue = mp.Queue()
        self.new_w_provider.cancel_join_thread()

        self.plot_process = mp.Process(name=f'rp_character_optimizer_plotter', target=self)

        self.inputs: np.ndarray = inputs
        self.outputs: np.ndarray = outputs

    def __call__(self):
        anim: FuncAnimation
        plotter = Plotter(self.inputs, self.outputs)

        def real_anim_func(frame: int) -> None:
            try:
                new_w: np.ndarray = self.new_w_provider.get_nowait()

            except queue.Empty:  # Queue is empty
                if self.end_event.is_set():
                    anim.event_source.stop()
                    plotter.render()
                    if self.output_dir:
                        plotter.save_plot(self.output_dir)
                    print('Finished plotting data')

                return

            plotter(frame, new_w)

        anim = FuncAnimation(plotter.fig, real_anim_func, interval=DEFAULT_ANIMATION_INTERVAL)
        plt.show()

    def start(self) -> None:
        self.plot_process.start()

    def publish(self, new_w: np.ndarray) -> None:
        self.new_w_provider.put(new_w)

    def close(self) -> None:
        self.end_event.set()

    def wait(self) -> None:
        self.close()
        self.plot_process.join()

    def kill(self):
        self.plot_process.terminate()


class Plotter:

    def __init__(self, inputs, outputs) -> None:

        self.fig = plt.figure()
        self.fig.tight_layout()
        plt.style.use('ggplot')

        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 3)

        self.ax = plt.gca()  # row 0, col 0 and 1
        # self.ax2 = plt.subplot(gs[0, 2])  # row 0, col 2
        # self.ax3 = plt.subplot(gs[1, 2])  # row 1, col 2
        # self.ax4 = plt.subplot(gs[1, :2])  # row 1, col 0 and 1

        self.positive_values: Dict[str, List[int]] = {
            'x': [],
            'y': []
        }
        self.negative_values: Dict[str, List[int]] = {
            'x': [],
            'y': []
        }

        for i in range(len(inputs)):
            if outputs[i] > 0:
                self.positive_values['x'].append(inputs[i][0])
                self.positive_values['y'].append(inputs[i][1])
            else:
                self.negative_values['x'].append(inputs[i][0])
                self.negative_values['y'].append(inputs[i][1])

    def __call__(self, frame: int, new_w: np.ndarray) -> None:
        # Always update data
        self.curr_w = np.copy(new_w)

        # Every render_step cycles (frames), render plot again
        self.render()

    def render(self):
        self.ax.clear()

        x = np.arange(-1.5, 1.5, 0.02)
        y = (- self.curr_w[1] * x - self.curr_w[0]) / self.curr_w[2]

        self.ax.plot(x, y, 'k')

        self.ax.scatter(self.positive_values['x'], self.positive_values['y'], color='red')
        self.ax.scatter(self.negative_values['x'], self.negative_values['y'], color='blue')

        plt.scatter(self.positive_values['x'], self.positive_values['y'], color='red')

        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Min, Max and Mean Fitness')


class NopAsyncPlotter(AsyncPlotter):

    def __init__(self) -> None:
        pass

    def __call__(self):
        pass

    def start(self) -> None:
        pass

    def publish(self, new_w: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def kill(self):
        pass


def _validate_plotter_params(plotter_params: Param) -> Param:
    return Config.validate_param(plotter_params, Schema({
        Optional('render', default=True): bool,
    }))


def get_plotter(plotter_params: Param, inputs: np.ndarray, outputs: np.ndarray) -> AsyncPlotter:
    if plotter_params['render']:
        return AsyncPlotter(inputs, outputs)
    else:
        return NopAsyncPlotter()
    # return AsyncPlotter(inputs, outputs)
