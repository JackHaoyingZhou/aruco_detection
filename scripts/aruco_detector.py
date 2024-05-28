import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from aruco_detection.msg import MarkerPose
from aruco_detection.msg import MarkerPoseArray
from sensor_msgs.msg import CameraInfo
import numpy as np
import tf

class ArucoDetector:
    def __init__(self, marker_length = 0.03, show_image=False):
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/depstech/image_raw", Image, self.image_callback)
        # self.image_sub = rospy.Subscriber("/realsense/image_raw", Image, self.image_callback)
        self.image_sub = rospy.Subscriber("/logitech/image_raw", Image, self.image_callback)
        self.poses_pub = rospy.Publisher("/aruco/marker_poses", MarkerPoseArray, queue_size=1)
        self.camera_matrix = None
        self.dist_coeffs = None
        # self.cam_info_sub = rospy.Subscriber("/depstech/camera_info", CameraInfo, self.cam_info_callback)
        # self.cam_info_sub = rospy.Subscriber("/realsense/camera_info", CameraInfo, self.cam_info_callback)
        self.cam_info_sub = rospy.Subscriber("/logitech/camera_info", CameraInfo, self.cam_info_callback)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()
        self.marker_length = marker_length
        self.show_img = show_image

    def cam_info_callback(self, msg):
        intrinsic_mtx = np.array(msg.K).reshape((3, 3))
        distortion_vec = np.array(msg.D).reshape((-1, 1))
        self.camera_matrix = intrinsic_mtx
        self.dist_coeffs = distortion_vec

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        marker_pose_array = MarkerPoseArray()

        if ids is not None:
            for i in range(len(ids)):
                corner = corners[i]
                marker_id = ids[i][0]
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, self.marker_length, self.camera_matrix, self.dist_coeffs)

                if self.show_img:
                    aruco.drawDetectedMarkers(cv_image, corners)
                    aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                    # Draw the ID of the marker
                    c = corner[0][0]
                    cv2.putText(cv_image, str(marker_id), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2, cv2.LINE_AA)

                marker_pose = MarkerPose()
                marker_pose.id = marker_id

                pose = Pose()
                pose.position.x = tvec[0][0][0]
                pose.position.y = tvec[0][0][1]
                pose.position.z = tvec[0][0][2]

                rot_matrix, _ = cv2.Rodrigues(rvec)
                # quat = self.rotation_matrix_to_quaternion(rot_matrix)
                quat = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((rot_matrix, [[0], [0], [0]])), [0, 0, 0, 1])))

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                marker_pose.pose = pose
                marker_pose_array.markers.append(marker_pose)

        self.poses_pub.publish(marker_pose_array)

        if self.show_img:
            cv2.imshow("Image Window", cv_image)
            k = cv2.waitKey(1)

            if k == 27 or k == ord('q'):
                cv2.destroyAllWindows()
                rospy.signal_shutdown("Image Window Closed")

    @staticmethod
    def rotation_matrix_to_quaternion(rot_matrix):
        q = np.zeros((4,))
        q[0] = np.sqrt(1.0 + rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]) / 2.0
        q[1] = (rot_matrix[2, 1] - rot_matrix[1, 2]) / (4.0 * q[0])
        q[2] = (rot_matrix[0, 2] - rot_matrix[2, 0]) / (4.0 * q[0])
        q[3] = (rot_matrix[1, 0] - rot_matrix[0, 1]) / (4.0 * q[0])
        return q

if __name__ == '__main__':
    rospy.init_node('aruco_detector', anonymous=True)
    aruco_detector = ArucoDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()
