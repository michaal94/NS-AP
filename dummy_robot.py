import zmq
context = zmq.Context()
socket_pose = context.socket(zmq.SUB)
socket_pose.setsockopt(zmq.SUBSCRIBE, b'')
socket_pose.connect("tcp://127.0.0.1:5557")
socket_gripper = context.socket(zmq.SUB)
socket_gripper.setsockopt(zmq.SUBSCRIBE, b'')
socket_gripper.connect("tcp://127.0.0.1:5556")

while True:
    #  Wait for next request from client
    message = socket_pose.recv()
    print("Pose: %s" % message)
    message = socket_gripper.recv()
    print("Gripper: %s" % message)
