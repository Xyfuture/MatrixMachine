from Desim.Core import SimModule,SimCoroutine,SimSession, SimTime
from Desim.module.FIFO import FIFO
from perf_tracer import PerfettoTracer
from typing import Dict, List, Optional

from .description import ComputeDie, Mapping, Chip, Tile




class SimComputeDie(SimModule):
    def __init__(self, compute_die: ComputeDie, tracer:PerfettoTracer, fifo_size: int = 10):
        super().__init__()

        self.tracer = tracer
        self.compute_die: ComputeDie = compute_die
        self.mapping: Optional[Mapping] = None
        self.task_queue:List[Tile] = []

        self.input_fifo: FIFO = FIFO(fifo_size)
        self.output_fifo: FIFO = FIFO(fifo_size)

        self.tracer_unit_name = f"ComputeDie-{self.compute_die.die_id}"
        self.track_info = self.tracer.register_module(self.tracer_unit_name)

        self.register_coroutine(self.input_process)
        self.register_coroutine(self.compute_process)
        self.register_coroutine(self.output_process)
    
    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        self.task_queue = self.mapping.placement[self.compute_die.die_id]

    def get_sim_time(self):
        return float(SimSession.sim_time.cycle)

    def input_process(self):
        input_track_info = self.tracer.register_track("Input",self.track_info)

        for i,task in enumerate(self.task_queue):
            with self.tracer.record_event(input_track_info, self.get_sim_time, f"Input-Task-{i}"):
                data_volume = task.batches * task.rows
                latency = int(data_volume // self.compute_die.config.input_bandwidth)
                SimModule.wait_time(SimTime(latency))
                self.input_fifo.write(i)
    def compute_process(self):
        compute_track_info = self.tracer.register_track("Compute",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.input_fifo.read()

            with self.tracer.record_event(compute_track_info, self.get_sim_time, f"Compute-Task-{i}"):
                compute_ops = task.batches * task.rows * task.cols
                compute_latency = int(compute_ops // (self.compute_die.config.compute_power * 10**3))

                memory_ops = task.rows * task.cols
                memory_latency = int(memory_ops // (self.compute_die.config.memory_bandwidth * 10**3))

                latency = max(compute_latency, memory_latency)

                SimModule.wait_time(SimTime(latency))
                self.output_fifo.write(i)

    def output_process(self):
        output_track_info = self.tracer.register_track("Output",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.output_fifo.read()
            with self.tracer.record_event(output_track_info, self.get_sim_time, f"Output-Task-{i}"):
                data_volume = task.batches * task.cols
                latency = int(data_volume // self.compute_die.config.output_bandwidth)
                SimModule.wait_time(SimTime(latency))


class SimChip:
    def __init__(self, chip: Chip, fifo_size: int = 10):
        
        SimSession.reset()
        SimSession.init()

        self.running_cycles = 0

        self.tracer = PerfettoTracer(100)
        self.chip: Chip = chip
        self.mapping: Optional[Mapping] = None

        self.sim_compute_dies: Dict[str, SimComputeDie] = {}
        for die_id, compute_die in self.chip.compute_dies.items():
            self.sim_compute_dies[die_id] = SimComputeDie(
                compute_die=compute_die,
                tracer=self.tracer,
                fifo_size=fifo_size
            )

    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        for die_id, sim_compute_die in self.sim_compute_dies.items():
            sim_compute_die.set_mapping(mapping)

    def run_sim(self):
        SimSession.scheduler.run()
        # self.tracer.save("trace.json")

        self.running_cycles = int(SimSession.sim_time.cycle)
        
    def get_running_cycles(self):
        return self.running_cycles  

    def save_trace_file(self, filename: str = "trace.json"):
        self.tracer.save(filename)



def simulate(chip:Chip, mapping: Mapping,save_trace:bool=False)->int:
    sim_chip = SimChip(chip)
    sim_chip.set_mapping(mapping)
    sim_chip.run_sim()
    if save_trace:
        sim_chip.save_trace_file("main_trace.json")
    return sim_chip.get_running_cycles()

    
