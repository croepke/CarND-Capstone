import numpy as np
import cv2
import tensorflow as tf
from styx_msgs.msg import TrafficLight
import rospy
from scipy.stats import mode

DETECTION_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self, is_site):
        self.is_site = is_site

        if self.is_site:
            PATH_TO_GRAPH = r'light_classification/model/ssd_site/frozen_inference_graph.pb'
            self.light_states= [(TrafficLight.UNKNOWN, 'UNKNOWN'), 
                                (TrafficLight.GREEN, 'GREEN'), 
                                (TrafficLight.RED, 'RED'), 
                                (TrafficLight.YELLOW, 'YELLOW'), 
                                (TrafficLight.UNKNOWN, 'UNKNOWN')]
        else:
            PATH_TO_GRAPH = r'light_classification/model/ssd_sim/frozen_inference_graph.pb'  
            self.light_states = [(TrafficLight.UNKNOWN, 'UNKNOWN'), 
                                 (TrafficLight.GREEN, 'GREEN'), 
                                 (TrafficLight.YELLOW, 'YELLOW'), 
                                 (TrafficLight.RED, 'RED'), 
                                 (TrafficLight.UNKNOWN, 'UNKNOWN')]

        self.detection_graph = self.load_graph(PATH_TO_GRAPH)
        if self.detection_graph:
            rospy.logwarn("is_site: {} - Traffic light graph loaded!".format(self.is_site))

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):   
        image = cv2.resize(image, (300, 300))    # use a smaller image to reduce processing time

        # Convert image from BGR Space to RGB for simulator model only
        if not self.is_site:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Expand image dimension for processing in tensorflow graph
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:
            scores, classes = sess.run([self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        if classes.any():
            score = scores[0]
            label = int(classes[0])
            if score > DETECTION_THRESHOLD:
                rospy.logwarn( "Light state: {}".format(self.light_states[label][1]) )
                return self.light_states[label][0]

        rospy.logwarn("Light state: UNKNOWN")
        return TrafficLight.UNKNOWN

    # Utility function
    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
