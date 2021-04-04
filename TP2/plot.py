from typing import List, Callable, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from schema import Schema

from config import Param, Config
from generation import Generation
from character import Character

import multiprocessing as mp

Figure = Any  # Stub for matplotlib
Animation = Callable[[Generation], None]
AnimationProvider = Callable[[Figure], Animation]


class AsyncPlotter:

    def __init__(self) -> None:
        self.new_gen_provider: mp.Queue = mp.Queue()
        self.new_gen_provider.cancel_join_thread()

        self.plot_process = mp.Process(name=f'rp_character_optimizer_plotter', target=self)

        self.running: bool = False

    def __call__(self):
        anim: FuncAnimation
        plotter = Plotter()

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

    def __init__(self) -> None:

        self.fig = plt.figure(figsize=(12, 8))

        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2)

        self.ax1 = plt.subplot(gs[0, 0])  # row 0, col 0
        self.ax2 = plt.subplot(gs[0, 1])  # row 0, col 1
        self.ax3 = plt.subplot(gs[1, :])  # entire row 1

        self.gens: List[int] = []
        self.min_fitness: List[float] = []
        self.mean_fitness: List[float] = []
        self.max_fitness: List[float] = []
        self.mean_diversity: List[float] = []
        self.diversity: Dict[str, List[float]] = {attr: [] for attr in Character.attr_list}

    def __call__(self, new_gen: Generation) -> None:
        self.gens.append(new_gen.gen_count)

        self._plot_min_max_fitness(self.ax1, new_gen)
        self._plot_mean_diversity(self.ax2, new_gen)
        self._plot_all_diversities(self.ax3, new_gen)

    def _plot_min_max_fitness(self, axis, new_gen: Generation) -> None:
        self.min_fitness.append(new_gen.get_min_fitness())
        self.max_fitness.append(new_gen.get_max_fitness())
        self.mean_fitness.append(new_gen.get_mean_fitness())

        axis.clear()

        l_min, = axis.plot(self.gens, self.min_fitness, 'r-')
        l_max, = axis.plot(self.gens, self.max_fitness, 'b-')
        l_mean, = axis.plot(self.gens, self.mean_fitness, 'k-')

        axis.set_xlabel('Generation')
        axis.set_ylabel('Fitness')
        axis.set_title('Min, Max and Mean Fitness')

        axis.legend([l_max, l_min, l_mean], ["Maximum Fitness", "Minimum Fitness", "Mean Fitness"])

    def _plot_mean_diversity(self, axis, new_gen: Generation) -> None:
        self.mean_diversity.append(new_gen.get_diversity().mean())

        axis.clear()

        l, = axis.plot(self.gens, self.mean_diversity, 'r-')

        axis.set_xlabel('Generation')
        axis.set_ylabel('Diversity')
        axis.set_title('Mean Diversity')

        axis.legend([l], ["Mean Diversity"])

    def _plot_all_diversities(self, axis, new_gen: Generation) -> None:
        diversity = np.array(new_gen.get_diversity())

        index: int = 0

        for attr in Character.attr_list:
            self.diversity[attr].append(diversity[index])
            index += 1

        axis.clear()

        l_agility, = axis.plot(self.gens, self.diversity['agility'], 'r-')
        l_endurance, = axis.plot(self.gens, self.diversity['endurance'], 'b-')
        l_experience, = axis.plot(self.gens, self.diversity['experience'], 'c-')
        l_height, = axis.plot(self.gens, self.diversity['height'], 'm-')
        l_strength, = axis.plot(self.gens, self.diversity['strength'], 'k-')
        l_vitality, = axis.plot(self.gens, self.diversity['vitality'], 'g-')

        axis.set_xlabel('Generation')
        axis.set_ylabel('Diversity')
        axis.set_title('All Diversities')

        # Estan bien ordenados - Clutch
        axis.legend([l_agility, l_endurance, l_experience, l_height, l_strength, l_vitality], Character.attr_list)


class NopAsyncPlotter(AsyncPlotter):

    def __init__(self) -> None:
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
        # Optional('plots', default=list): And(list, Plotter.supported_plots)
    }))


def get_plotter(plotter_params: Param) -> AsyncPlotter:
    plotter_params = _validate_plotter_params(plotter_params)
    if plotter_params['render']:
        return AsyncPlotter()
    else:
        return NopAsyncPlotter()
