from Desim.Core import SimModule,SimCoroutine,SimSession, SimTime
from Desim.module.FIFO import FIFO
from perf_tracer import PerfettoTracer
from typing import Dict

from .description import ComputeDie, Mapping, Chip




class SimComputeDie(SimModule):
    def __init__(self, compute_die: ComputeDie, mapping: Mapping, fifo_size: int = 10):
        super().__init__()

        self.tracer = PerfettoTracer.get_global_tracer()
        self.compute_die: ComputeDie = compute_die
        self.mapping: Mapping = mapping

        self.input_fifo: FIFO = FIFO(fifo_size)
        self.output_fifo: FIFO = FIFO(fifo_size)

        self.task_queue = self.mapping.placement[self.compute_die.die_id]

        self.tracer_unit_name = f"ComputeDie-{self.compute_die.die_id}"

        self.track_info = self.tracer.register_module(self.tracer_unit_name)

        self.register_coroutine(self.input_process)
        self.register_coroutine(self.compute_process)
        self.register_coroutine(self.output_process)

    def get_sim_time(self):
        return float(SimSession.sim_time.cycle)

    def input_process(self):
        input_track_info = self.tracer.register_track("Input",self.track_info)

        for i,task in enumerate(self.task_queue):
            with self.tracer.record_event(input_track_info, self.get_sim_time, f"Input-Task-{i}"):
                latency = int(task.rows // self.compute_die.config.input_bandwidth)
                SimModule.wait_time(SimTime(latency))
                self.input_fifo.write(i)
    def compute_process(self):
        compute_track_info = self.tracer.register_track("Compute",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.input_fifo.read()

            with self.tracer.record_event(compute_track_info, self.get_sim_time, f"Compute-Task-{i}"):
                latency = int(task.rows * task.cols // (self.compute_die.config.compute_power * 10**3))
                SimModule.wait_time(SimTime(latency))
                self.output_fifo.write(i)

    def output_process(self):
        output_track_info = self.tracer.register_track("Output",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.output_fifo.read()
            with self.tracer.record_event(output_track_info, self.get_sim_time, f"Output-Task-{i}"):
                latency = int(task.cols // self.compute_die.config.output_bandwidth)
                SimModule.wait_time(SimTime(latency))


class SimChip:
    def __init__(self, chip: Chip, mapping: Mapping, fifo_size: int = 10):
        
        SimSession.init()

        PerfettoTracer.init_global_tracer()
        self.tracer = PerfettoTracer.get_global_tracer()
        self.chip: Chip = chip
        self.mapping: Mapping = mapping

        self.sim_compute_dies: Dict[str, SimComputeDie] = {}
        for die_id, compute_die in self.chip.compute_dies.items():
            self.sim_compute_dies[die_id] = SimComputeDie(
                compute_die=compute_die,
                mapping=mapping,
                fifo_size=fifo_size
            )

    def run_sim(self):
        SimSession.scheduler.run() 
        self.tracer.save("trace.json")  



