import paho.mqtt.client as mqtt

def create_mqtt(broker,port=1883,id='front'):
    client = mqtt.Client(id)
    client.connect(broker,port,60)
    client.loop_start()

def subscribe(client,topic):
    client.subscribe(topic)

def post():
    pass