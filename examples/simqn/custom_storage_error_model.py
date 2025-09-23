import numpy as np

from mqns.entity.memory.memory import QuantumMemory
from mqns.entity.node.qnode import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.simulator.simulator import Simulator


class ErrorEntanglement(WernerStateEntanglement):
    def store_error_model(self, t: float, **kwargs):
        # storage error will reduce the fidelity
        t_coh = kwargs.get("t_coh", 1)
        self.w = self.w * np.exp(-1 / t_coh)
        print("xxx")


n1 = QNode("n1")

# memory error attributions: coherence time is 1 second
m3 = QuantumMemory("m3", capacity=10, decoherence_rate=0.2, store_error_model_args={"t_coh": 1})
n1.add_memory(m3)

s = Simulator(0, 10, 1000)
n1.install(s)

# generate an entanglement and store it
epr1 = ErrorEntanglement(name="epr1")
m3.write(epr1)

# after a while, the fidelity will drop
epr2 = m3.read(epr1)
print(epr2.fidelity)
