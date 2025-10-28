import json, time, paho.mqtt.client as mqtt

class MQTTPublisher:
    def __init__(self, host='localhost', port=1883, topic='edgevision/events', status_topic='edgevision/status'):
        self.topic = topic; self.status_topic = status_topic
        self.client = mqtt.Client()
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()
        self.publish_status('online')

    def publish_status(self, status):
        self.client.publish(self.status_topic, json.dumps({'status': status, 'ts': time.time()}), retain=True)

    def publish_event(self, payload: dict):
        self.client.publish(self.topic, json.dumps(payload))

    def close(self):
        try:
            self.publish_status('offline')
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
