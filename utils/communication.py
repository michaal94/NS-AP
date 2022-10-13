"""
Created on Wed May 26 14:56:16 2021

@author: syxtreme
"""

import zmq
from threading import Thread, RLock
from warnings import warn
import cloudpickle as cpl
from zmq.backend import zmq_errno
from zmq.utils.strtypes import asbytes
import time


class ParamClient():
    DEFAULT_TIMEOUT = 10000  # milliseconds

    def __init__(self, start_port=25652, addr="127.0.0.1", protocol="tcp"):
        """ Creates a client for parameters. The client can subscribe to various parameters
        and then it will receive updates about changes to these parameters.
        The parameters can be accessed as if they were properties of this client.
        ***WARNING***
        The property voodoo does changes to the base class itself. That means,
        using more clients in the same script will be buggy - all instances will have the same
        properties. Although, this should be ok in most cases, it is recommended to find another way.

        Args:
            start_port (int, optional): A port where the parameters server operates. This is a port on which
            only the publisher will send parameter updates. Parameter changes and definitions/declarations
            will be done over ports with numbers +3 and +7 higher. Defaults to 25652.
            addr (str, optional): IP address on which the server operates. Defaults to "127.0.0.1".
            protocol (str, optional): Protocol to use for communication.
        """
        self.__context = zmq.Context()

        self._change_lock = RLock()
        self._define_lock = RLock()

        # subscirbe to parameter changes
        self.__addr_pub = f"{protocol}://{addr}:{str(start_port)}"
        self._subscriber = self.__context.socket(zmq.SUB)
        self._subscriber.connect(self.__addr_pub)
        self._subscriber.setsockopt(zmq.RCVTIMEO, self.DEFAULT_TIMEOUT)

        # socket for param change requests
        self.__addr_ch = f"{protocol}://{addr}:{str(start_port + 3)}"
        self._changer = self.__context.socket(zmq.REQ)
        self._changer.connect(self.__addr_ch)

        # socket for param definition
        self.__addr_def = f"{protocol}://{addr}:{str(start_port + 7)}"
        self._definer = self.__context.socket(zmq.REQ)
        self._definer.connect(self.__addr_def)

        self._params = {}
        self.active = True

        self.poller_thread = Thread(target=self._poll, daemon=True)
        self.poller_thread.start()

    def get_all(self):
        return self._params

    def wait_for_param(self, param, timeout=0):
        msg = self.wait_receive_param(param, timeout)
        if msg is None:
            return False
        return True

    def wait_receive_param(self, param, timeout=0):
        self._subscriber.setsockopt(zmq.SUBSCRIBE, param.encode('utf-8'))
        if timeout > 0:
            self._subscriber.setsockopt(zmq.RCVTIMEO, timeout)
        msg = None
        while True:
            try:
                rcv_param, data = self._subscriber.recv_multipart()
            except zmq.error.Again:
                print(f"Paramater {param} request timed out!")
                return None
            if rcv_param.decode() == param:
                msg = cpl.loads(data)
                print(param, msg)
                break
        self._subscriber.setsockopt(zmq.UNSUBSCRIBE, param.encode('utf-8'))
        if timeout > 0:
            self._subscriber.setsockopt(zmq.RCVTIMEO, self.DEFAULT_TIMEOUT)
        if msg is not None:
            return msg

    def subscribe(self, param):
        """Subscribe to parameter updates. This does not guarantee that the parameter
        exists and has any value. This only tells the server that this client
        wants to be notified if a parameter with the specified name changes.
        At the begining, the value of the parameter will be "None" and will remain so
        until this client receives value update for the parameter. This will occur
        rather quickly if the parameter exists on the server but this is not known
        at the time this method is called. This should be fine, just be aware of it.

        Args:
            param (str): The name of the parameter
        """
        if hasattr(self, param):
            return

        self._params[param] = None
        setattr(self.__class__, param, property(
                lambda self, param=param: self._params[param],
                lambda self, value, param=param: self._set_param(param, value)
            ))
        self._subscriber.setsockopt(zmq.SUBSCRIBE, param.encode('utf-8'))

    def declare(self, param, default_value=None, type=None, description=""):
        """Subsribes to parameter updates. This does roughly the same as
        subscribe but presents a default value to fill in if the parameter
        does not exist on the server. Use this if you want to always start with some
        (not None) value but you don't want to overwrite any existing values.
        E.g., you need the param to be False at the beginning but remain True
        if it was set to True by other client (with the subscribe method,
        it would not be False but None, if it was not defined, yet).

        Args:
            param (str): The name of the parameter
            default_value (any, optional): Default value for the parameter if it was not
            defined, yet. Defaults to None.
        """
        self.__send_definition(param, default_value, 0)

    def define(self, param, value, type=None, description=""):
        """Subscribes to parameter updates. Unlike subscribe or declare, this method always
        overwrites the current parameter value. That is, whether the parameter exists on the server
        or not, after calling this method, it will have value set to "value".
        Use this method for clients that should "own" the control over this parameter.

        Args:
            param (str): The name of the parameter.
            value (any): The starting or new value of the parameter. The parameter is always
            set to this value upon definition.
        """
        self.__send_definition(param, value, 1)

    def destroy(self):
        """Stops the updates and closes all connection.
        """
        self.active = False
        self._subscriber.close()
        self._changer.close()
        self._definer.close()

    def __send_definition(self, param, value, overwrite):
        self._define_lock.acquire()
        self._definer.send_multipart([param.encode('utf-8'), cpl.dumps(value), asbytes(chr(overwrite))])
        response = self._definer.recv().decode()
        if response == "false":
            raise AttributeError(f"Could not declare the parameter {param} (value={value}).")
        self._define_lock.release()
        self.subscribe(param)

    def _set_param(self, param, value):
        self._change_lock.acquire()
        self._changer.send_multipart([param.encode('utf-8'), cpl.dumps(value)])
        response = self._changer.recv().decode()
        # print(response)
        if response == "true":
            self._params[param] = value
        self._change_lock.release()

    def _poll(self):
        while self.active:
            try:
                param, msg = self._subscriber.recv_multipart()
            except zmq.Again:
                continue
            # print(param, msg)
            self._params[param.decode()] = cpl.loads(msg)


class ParamSubscriber(ParamClient):
    """The same as ParamClient, but calls a function.
    when receiving a parameter.
    """

    def __init__(self, start_port=25652, addr="127.0.0.1", protocol="tcp"):
        super().__init__(start_port, addr, protocol)
        self._cb = lambda para, msg: None  # default "empty" callback function

    def set_callback(self, cb):
        self._cb = cb
    
    def _poll(self):
        while self.active:
            try:
                rcv_param, data = self._subscriber.recv_multipart()
            except zmq.Again:
                continue
            msg = cpl.loads(data)
            param = rcv_param.decode()
            # print(param, msg)
            self._params[param] = msg
            self._cb(param, msg)


class UniversalParamClient(ParamSubscriber):

    def __init__(self, start_port=25652, addr="127.0.0.1", protocol="tcp"):
        super().__init__(start_port, addr, protocol)

        # Subscribe to all
        self._subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    def attach(self, param, default_value=None, force=False):
        # just adds the param to the class
        self._params[param] = default_value
        setattr(self.__class__, param, property(
                lambda self, param=param: self._params[param],
                lambda self, value, param=param: self._set_param(param, value)
            ))
        if force:
            self._definer.send_multipart([param.encode('utf-8'), cpl.dumps(default_value), asbytes(chr(0))])
            response = self._definer.recv().decode()
            if response == "false":
                raise AttributeError(f"Could not declare the parameter {param} (value={default_value}).")

    def _poll(self):
        while self.active:
            try:
                rcv_param, data = self._subscriber.recv_multipart()
            except zmq.Again:
                continue
            msg = cpl.loads(data)
            param = rcv_param.decode()
            # print(param, msg)
            self._params[param] = msg
            self._cb(param, msg)