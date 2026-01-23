import numpy as np

from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.error import CoherentErrorModel
from mqns.network.protocol import BB84RecvApp, BB84SendApp
from mqns.simulator import Simulator


def test_bb84_protocol():
    light_speed = 299792.458  # km/s
    length = 1  # km

    n1 = QNode(name="n1")
    n2 = QNode(name="n2")

    qlink = QuantumChannel(
        name="l1",
        length=length,
        delay=length / light_speed,
        drop_rate=1 - np.exp(-length / 50),  # 0.2 db/km
        transfer_error=CoherentErrorModel(length=length),
    )

    clink = ClassicChannel(name="c1", delay=length / light_speed)

    n1.add_cchannel(clink)
    n2.add_cchannel(clink)
    n1.add_qchannel(qlink)
    n2.add_qchannel(qlink)

    sp = BB84SendApp(n2, qlink, clink, send_rate=1000)
    rp = BB84RecvApp(n1, qlink, clink)
    n1.add_apps(sp)
    n2.add_apps(rp)

    s = Simulator(0, 15, accuracy=10_000_000_000, install_to=(n1, n2))
    s.run()
    print("BB84SendApp", sp.fail_number, len(sp.successful_key))
    print("BB84RecvApp", rp.fail_number, len(rp.successful_key))
    assert len(sp.successful_key) > 0
    assert sp.successful_key == rp.successful_key
