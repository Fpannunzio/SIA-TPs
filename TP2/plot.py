from typing import List, Callable, Any, Collection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from schema import Schema, And, Optional

from config import Param, Config
from generation import Generation

import multiprocessing as mp

Figure = Any  # Stub for matplotlib
Animation = Callable[[Generation], None]
AnimationProvider = Callable[[Figure], Animation]


class AsyncPlotter:

    def __init__(self, plots: Collection[str]) -> None:
        self.plots = plots
        self.new_gen_provider: mp.Queue = mp.Queue()
        self.new_gen_provider.cancel_join_thread()

        self.plot_process = mp.Process(name=f'rp_character_optimizer_plotter', target=self)

        self.running: bool = False

    def __call__(self):
        anim: FuncAnimation
        plotter = Plotter(self.plots)

        def real_anim_func(frame: int) -> None:
            if plotter.gens and self.new_gen_provider.empty():
                anim.event_source.stop()
                return

            plotter(self.new_gen_provider.get())

        anim = FuncAnimation(plotter.fig, real_anim_func, interval=100)
        plt.show()

    def start(self) -> None:
        self.plot_process.start()

    def is_running(self):
        return self.running

    def publish(self, new_gen: Generation) -> None:
        self.new_gen_provider.put(new_gen)

    def close(self) -> None:
        self.new_gen_provider.close()

    def wait(self) -> None:
        self.plot_process.join()

    def kill(self):
        self.plot_process.terminate()


class Plotter:

    supported_plots: List[str] = ['min_fitness', 'max_fitness']

    def __init__(self, plots: Collection[str]) -> None:
        self.min_fitness = 'min_fitness' in plots
        self.max_fitness = 'min_fitness' in plots

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8)) #2 rows y 2 cols --> 4 graphs

        self.gens: List[int] = []
        self.min_fitness: List[float] = []
        self.mean_fitness: List[float] = []
        self.max_fitness: List[float] = []

    def __call__(self, new_gen: Generation) -> None:
        self.gens.append(new_gen.gen_count)
        self.min_fitness.append(new_gen.get_min_fitness())
        # mean_fitness.append(gen.mean_fitness())
        self.max_fitness.append(new_gen.get_max_fitness())

        self.ax1.clear()
        self.ax2.clear()

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness real-time")

        l1, = self.ax1.plot(self.gens, self.min_fitness, 'r-')
        # l2, = ax.plot(gens, mean_fitness, 'g-')
        l3, = self.ax2.plot(self.gens, self.max_fitness, 'b-')

        # plt.legend([l3, l2, l1], ["Maximum Fitness", "Mean Fitness", "Minimum Fitness"])
        plt.legend([l3, l1], ["Maximum Fitness", "Minimum Fitness"])


class NopAsyncPlotter(AsyncPlotter):

    def __init__(self, plots: Collection[str]) -> None:
        pass

    def __call__(self):
        pass

    def start(self) -> None:
        pass

    def is_running(self):
        pass

    def publish(self, new_gen: Generation) -> None:
        pass

    def close(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def kill(self):
        pass


def _validate_plotter_params(plotter_params: Param) -> Param:
    return Config.validate_param(plotter_params, Schema({
        'render': bool,
        Optional('plots', default=list): And(list, Plotter.supported_plots)
    }))


def get_plotter(plotter_params: Param) -> AsyncPlotter:
    plotter_params = _validate_plotter_params(plotter_params)
    if plotter_params['render'] and plotter_params['plots']:
        return AsyncPlotter(plotter_params['plots'])
    else:
        return NopAsyncPlotter(plotter_params['plots'])
