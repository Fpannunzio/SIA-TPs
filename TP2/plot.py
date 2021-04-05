import queue
from typing import List, Callable, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import typing
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from schema import Schema, Optional, And

from config import Param, Config
from generation import Generation
from character import Character

import multiprocessing as mp

Figure = Any  # Stub for matplotlib
Animation = Callable[[Generation], None]
AnimationProvider = Callable[[Figure], Animation]

# TODO: Definir mejores defaults
DEFAULT_ANIMATION_INTERVAL: int = 50  # Data is processed every 100 ms
DEFAULT_RENDER_STEP: int = 10  # Plot is re-rendered every ANIMATION_INTERVAL*render_step ms


class AsyncPlotter:

    def __init__(self, render_step: int) -> None:
        self.end_event = mp.Event()
        self.new_gen_provider: mp.Queue = mp.Queue()
        self.new_gen_provider.cancel_join_thread()

        self.plot_process = mp.Process(name=f'rp_character_optimizer_plotter', target=self)

        self.render_step = render_step

    def __call__(self):
        anim: FuncAnimation
        plotter = Plotter(self.render_step)

        def real_anim_func(frame: int) -> None:
            try:
                new_gen: Generation = self.new_gen_provider.get_nowait()

            except queue.Empty:  # Queue is empty
                if self.end_event.is_set():
                    anim.event_source.stop()
                    plotter.render()
                    print('Finished plotting data')

                return

            plotter(frame, new_gen)

        anim = FuncAnimation(plotter.fig, real_anim_func, interval=DEFAULT_ANIMATION_INTERVAL)
        plt.show()

    def start(self) -> None:
        self.plot_process.start()

    def publish(self, new_gen: Generation) -> None:
        self.new_gen_provider.put(new_gen)

    def close(self) -> None:
        self.end_event.set()

    def wait(self) -> None:
        self.close()
        self.plot_process.join()

    def kill(self):
        self.plot_process.terminate()


class Plotter:

    def __init__(self, render_step: int) -> None:

        self.render_step = render_step
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.tight_layout()
        plt.style.use('ggplot')

        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 3)

        self.ax1 = plt.subplot(gs[0, :2])  # row 0, col 0 and 1
        self.ax2 = plt.subplot(gs[0, 2])   # row 0, col 2
        self.ax3 = plt.subplot(gs[1, 2])   # row 1, col 2
        self.ax4 = plt.subplot(gs[1, :2])  # row 1, col 0 and 1

        self.gens: List[int] = []

        self.min_fitness: List[float] = []
        self.mean_fitness: List[float] = []
        self.max_fitness: List[float] = []

        self.best_char_stats: Dict[str, List[float]] = {
            'strength': [],
            'agility': [],
            'experience': [],
            'endurance': [],
            'vitality': [],
            'height': [],
            'attack': [],
            'defence': []
        }

        # self.mean_diversity: List[float] = []

        self.diversity: Dict[str, List[float]] = {attr: [] for attr in Character.attr_list}

    def __call__(self, frame: int, new_gen: Generation) -> None:
        # Always update data
        self.add_new_gen(new_gen)

        # Every render_step cycles (frames), render plot again
        if frame % self.render_step == 0:
            self.render()

    def add_new_gen(self, new_gen: Generation):
        self.gens.append(new_gen.gen_count)

        self._update_min_max_fitness(new_gen)
        self._update_best_char_stats(new_gen)
        self._update_all_diversities(new_gen)

    def render(self):
        self._plot_min_max_fitness(self.ax1)
        self._plot_best_character_stats(self.ax2, self.ax3)
        self._plot_diversity(self.ax4)

    def _update_min_max_fitness(self, new_gen: Generation):
        self.min_fitness.append(new_gen.get_min_fitness())
        self.max_fitness.append(new_gen.get_max_fitness())
        self.mean_fitness.append(new_gen.get_mean_fitness())

    def _update_best_char_stats(self, new_gen: Generation):
        best_char: Character = new_gen.get_best_character()

        self.best_char_stats['strength'].append(best_char.get_strength())
        self.best_char_stats['agility'].append(best_char.get_agility())
        self.best_char_stats['experience'].append(best_char.get_experience())
        self.best_char_stats['endurance'].append(best_char.get_endurance())
        self.best_char_stats['vitality'].append(best_char.get_vitality())
        self.best_char_stats['height'].append(best_char.height)
        self.best_char_stats['attack'].append(best_char.get_attack())
        self.best_char_stats['defence'].append(best_char.get_defence())

    def _update_all_diversities(self, new_gen: Generation):
        diversity = np.array(new_gen.get_diversity())

        for idx, attr in enumerate(Character.attr_list):
            self.diversity[attr].append(diversity[idx])

    def _plot_min_max_fitness(self, axis) -> None:
        axis.clear()

        l_min, = axis.plot(self.gens, self.min_fitness, 'r-')
        l_max, = axis.plot(self.gens, self.max_fitness, 'b-')
        l_mean, = axis.plot(self.gens, self.mean_fitness, 'k-')

        axis.set_xlabel('Generation')
        axis.set_ylabel('Fitness')
        axis.set_title('Min, Max and Mean Fitness')

        axis.legend([l_max, l_min, l_mean], ["Maximum Fitness", "Minimum Fitness", "Mean Fitness"])

    def _plot_best_character_stats(self, base_stats_axis, calculated_stats_axis) -> None:
        base_stats_axis.clear()
        calculated_stats_axis.clear()

        l_agility, = base_stats_axis.plot(self.gens, self.best_char_stats['agility'], 'r-')
        l_endurance, = base_stats_axis.plot(self.gens, self.best_char_stats['endurance'], 'b-')
        l_experience, = base_stats_axis.plot(self.gens, self.best_char_stats['experience'], 'c-')
        l_height, = base_stats_axis.plot(self.gens, self.best_char_stats['height'], 'm-')
        l_strength, = base_stats_axis.plot(self.gens, self.best_char_stats['strength'], 'k-')
        l_vitality, = base_stats_axis.plot(self.gens, self.best_char_stats['vitality'], 'g-')

        l_attack, = calculated_stats_axis.plot(self.gens, self.best_char_stats['attack'], 'r-')
        l_defence, = calculated_stats_axis.plot(self.gens, self.best_char_stats['defence'], 'b-')

        base_stats_axis.set_xlabel('Generation')
        base_stats_axis.set_ylabel('Stats (same colors as Genetic Diversity)')
        base_stats_axis.set_title('Stats of Best Individual of Generation')

        calculated_stats_axis.set_xlabel('Generation')
        calculated_stats_axis.set_ylabel('Attack & Defence')
        calculated_stats_axis.set_title('Attack and Defence of Best Individual of Generation')

        # base_stats_axis.legend([l_agility, l_endurance, l_experience, l_height, l_strength, l_vitality],
        #                        ['agility', 'endurance', 'experience', 'height', 'strength', 'vitality'])

        calculated_stats_axis.legend([l_attack, l_defence], ['attack', 'defence'])

    def _plot_diversity(self, axis) -> None:
        axis.clear()

        l_agility, = axis.plot(self.gens, self.diversity['agility'], 'r-')
        l_endurance, = axis.plot(self.gens, self.diversity['endurance'], 'b-')
        l_experience, = axis.plot(self.gens, self.diversity['experience'], 'c-')
        l_height, = axis.plot(self.gens, self.diversity['height'], 'm-')
        l_strength, = axis.plot(self.gens, self.diversity['strength'], 'k-')
        l_vitality, = axis.plot(self.gens, self.diversity['vitality'], 'g-')

        axis.set_xlabel('Generation')
        axis.set_ylabel('Diversity')
        axis.set_title('Genetic Diversity')

        # Estan bien ordenados - Clutch
        axis.legend([l_agility, l_endurance, l_experience, l_height, l_strength, l_vitality], Character.attr_list)


class NopAsyncPlotter(AsyncPlotter):

    def __init__(self, render_step: int) -> None:
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
        Optional('render', default=True): bool,
        Optional('process_gen_interval', default=DEFAULT_ANIMATION_INTERVAL): And(int, lambda ms: ms > 0),
        Optional('step', default=DEFAULT_RENDER_STEP): And(int, lambda step: step > 0)
    }))


def get_plotter(plotter_params: Param) -> AsyncPlotter:
    plotter_params = _validate_plotter_params(plotter_params)
    if plotter_params['render']:
        return AsyncPlotter(plotter_params['step'])
    else:
        return NopAsyncPlotter(plotter_params['step'])
