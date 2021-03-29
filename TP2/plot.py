from typing import List

# TODO(charlar): Para que ande con PyCharm - Lo queremos?
import matplotlib; matplotlib.use("TkAgg")
from generation import Generation
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self) -> None:
        self.min_fitness: List[float] = []
        self.generation: List[int] = []

        plt.style.use('ggplot')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.show()

    def publish(self, new_generation: Generation):
        self.min_fitness.append(new_generation.get_min_fitness())
        self.generation.append(new_generation.gen_count)

        self.ax.plot(self.generation, self.min_fitness, color='b')

        self.fig.canvas.draw()

        self.ax.set_xlim(left=max(0, new_generation.gen_count - 50), right=new_generation.gen_count + 50)
        self.fig.canvas.flush_events()
        plt.pause(0.01)


