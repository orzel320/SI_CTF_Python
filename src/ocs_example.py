from osc_messenger import OSCMessenger

osc = OSCMessenger(listen_port=8000, unreal_port=8001)


def on_init(packet_id: int, b1_x: float, b1_y: float, b2_x: float, b2_y: float) -> None:
    # initialization logic
    osc.send_ok(packet_id)


def on_step(
    packet_id: int,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
    p1_flag: bool,
    p2_flag: bool,
) -> None:
    # AI step calculation
    osc.send_step(packet_id, p1_x, p1_y, p2_x, p2_y)


def on_end(packet_id: int, winner_id: int) -> None:
    # end-of-game logic
    osc.send_ok(packet_id)


osc.register_handler("init", on_init)
osc.register_handler("step", on_step)
osc.register_handler("end", on_end)

osc.start()
