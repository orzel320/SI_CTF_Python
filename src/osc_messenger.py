"""Standalone OSC communication module for Unreal Engine 5 and Python."""

from typing import Any, Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


class OSCMessenger:
    """Reads and sends OSC messages between Unreal Engine 5 and Python.

    Attributes:
        client (SimpleUDPClient): UDP client for sending messages to UE5.
        out_address (str): Target OSC address for outgoing messages.
        handlers (dict[str, Callable]): Registered callbacks for incoming flags.
        server (BlockingOSCUDPServer): UDP server listening for incoming messages.
    """

    def __init__(
        self,
        listen_ip: str = "127.0.0.1",
        listen_port: int = 8000,
        unreal_ip: str = "127.0.0.1",
        unreal_port: int = 8001,
        in_address: str = "/*",
        out_address: str = "/ctf",
    ) -> None:
        """Initializes the OSC UDP server and client.

        Args:
            listen_ip (str): IP address for the Python server to bind to.
            listen_port (int): Port for the Python server to listen on.
            unreal_ip (str): IP address of the Unreal Engine OSC listener.
            unreal_port (int): Port of the Unreal Engine OSC listener.
            in_address (str): OSC address pattern to listen on (default '/*').
            out_address (str): OSC address pattern for outgoing packets.
        """
        self.client = SimpleUDPClient(unreal_ip, unreal_port)
        self.out_address = out_address
        self.handlers: dict[str, Callable[..., None]] = {}

        dispatcher = Dispatcher()
        dispatcher.map(in_address, self._on_message)
        self.server = BlockingOSCUDPServer((listen_ip, listen_port), dispatcher)

    def register_handler(self, flag: str, callback: Callable[..., None]) -> None:
        """Registers a callback function for a specific message flag.

        Args:
            flag (str): Message type flag (e.g., 'init', 'step', 'end').
            callback (Callable[..., None]): Function called with (packet_id, *data).
        """
        self.handlers[flag] = callback

    def _on_message(self, address: str, flag: str, packet_id: int, *data: Any) -> None:
        """Routes incoming OSC packets to the corresponding registered handler.

        Args:
            address (str): Incoming OSC address path.
            flag (str): Packet type identifier ('init', 'step', or 'end').
            packet_id (int): Sequence identifier of the incoming packet.
            *data (Any): Remaining payload arguments sent by Unreal Engine.
        """
        handler = self.handlers.get(flag)
        if handler:
            handler(packet_id, *data)

    def send(self, flag: str, packet_id: int, *data: Any) -> None:
        """Sends a formatted OSC packet to Unreal Engine.

        Args:
            flag (str): Response flag ('ok' or 'step').
            packet_id (int): ID of the packet being acknowledged.
            *data (Any): Additional payload values to include in the packet.
        """
        self.client.send_message(self.out_address, [flag, int(packet_id), *data])

    def send_ok(self, packet_id: int) -> None:
        """Sends an 'ok' acknowledgment packet to Unreal Engine.

        Args:
            packet_id (int): ID of the received packet to acknowledge.
        """
        self.send("ok", packet_id)

    def send_step(
        self,
        packet_id: int,
        p1_dest_x: float,
        p1_dest_y: float,
        p2_dest_x: float,
        p2_dest_y: float,
    ) -> None:
        """Sends a 'step' response with destination coordinates to Unreal Engine.

        Args:
            packet_id (int): ID of the received 'step' packet.
            p1_dest_x (float): Target X coordinate for Player 1.
            p1_dest_y (float): Target Y coordinate for Player 1.
            p2_dest_x (float): Target X coordinate for Player 2.
            p2_dest_y (float): Target Y coordinate for Player 2.
        """
        self.send("step", packet_id, p1_dest_x, p1_dest_y, p2_dest_x, p2_dest_y)

    def start(self) -> None:
        """Starts the blocking OSC server loop."""
        self.server.serve_forever()

    def stop(self) -> None:
        """Stops the OSC server and releases the socket."""
        self.server.shutdown()
        self.server.server_close()
