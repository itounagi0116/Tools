import cv2
import numpy as np
import mediapipe as mp
import openvr
import time  #  Import time only, not json or socket (for initial simplification)
from pythonosc import udp_client

class VMCTrackingSystem:
    def __init__(self):
        # Initialize OpenVR
        self.vr_system = openvr.init(openvr.VRApplication_Background)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize all trackers
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize VMC Protocol client
        self.vmc_client = udp_client.SimpleUDPClient("127.0.0.1", 39539)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Face landmark indices for eye tracking
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

        # Face landmark indices for expressions
        self.MOUTH_INDICES = [61, 291]  # Upper and lower lip
        self.BROW_INDICES = [65, 295]  # Left and right eyebrow

        #OpenVR Tracker Initialization
        self.openvr_trackers = {} #Dictionary to keep track of trackers


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process all tracking data
        pose_results = self.pose.process(frame_rgb)
        hands_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        tracking_data = {
            "pose": None,
            "hands": [],
            "face": None,
            "expressions": {}
        }

        # Process pose data
        if pose_results.pose_landmarks:
            tracking_data["pose"] = self.process_pose_landmarks(pose_results.pose_landmarks.landmark)

        # Process hands data
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_data = self.process_hand_landmarks(hand_landmarks.landmark)
                tracking_data["hands"].append(hand_data)

        # Process face and expression data
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            tracking_data["face"] = self.process_face_landmarks(face_landmarks)
            tracking_data["expressions"] = self.process_expressions(face_landmarks)

        # Send all tracking data via VMC Protocol
        self.send_vmc_data(tracking_data)

        # Update OpenVR trackers
        self.update_openvr_trackers(tracking_data)

        return frame

    def process_pose_landmarks(self, landmarks):
        """Process pose landmarks"""
        bone_data = {}

        # Define a mapping of landmark indices to bone names.  This is a *subset*
        # of the bones VMC Protocol supports, focusing on the ones easily
        # derived from MediaPipe's pose estimation.  Critically, this includes
        # the Hips, Spine, Chest, Neck, Head, and limb bones.
        bone_mapping = {
             0: "Hips",  # Using Nose as a proxy for hips.  This isn't ideal.
            11: "LeftUpperArm",
            13: "LeftLowerArm",
            15: "LeftHand",  # Wrist
            12: "RightUpperArm",
            14: "RightLowerArm",
            16: "RightHand", # Wrist
            23: "LeftUpperLeg",
            25: "LeftLowerLeg",
            27: "LeftFoot",  # Ankle
            24: "RightUpperLeg",
            26: "RightLowerLeg",
            28: "RightFoot",  # Ankle
        }

        # Add additional bones based on calculations
        bone_data["Spine"] = self.calculate_midpoint(landmarks, 11, 12, 23, 24) # Calculate spine
        bone_data["Chest"] = self.calculate_midpoint(landmarks, 11, 12) #Calculate Chest
        bone_data["Neck"] = self.calculate_midpoint(landmarks, 11, 12)  # Neck (same as chest for now, improve later)
        bone_data["Head"] = {"position": [landmarks[0].x, landmarks[0].y, landmarks[0].z], "rotation":[0,0,0,1]} #Added Head


        for landmark_index, bone_name in bone_mapping.items():
            landmark = landmarks[landmark_index]
            bone_data[bone_name] = {
                "position": [landmark.x, landmark.y, landmark.z],
                "rotation": [0, 0, 0, 1]  # Placeholder, calculate real rotations later
            }

        return {"bones": bone_data}


    def calculate_midpoint(self, landmarks, *indices):
        """Calculates the midpoint between multiple landmarks."""
        points = [np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) for i in indices]
        midpoint = np.mean(points, axis=0)
        return {"position": midpoint.tolist(), "rotation": [0, 0, 0, 1]}



    def process_hand_landmarks(self, landmarks):
        """Process hand landmarks for finger tracking"""
        finger_data = {
            "thumb": [],
            "index": [],
            "middle": [],
            "ring": [],
            "pinky": []
        }

        # MediaPipe hand landmark indices for each finger
        finger_indices = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        # Extract positions and calculate rotations for each finger joint
        for finger_name, indices in finger_indices.items():
            for idx in indices:
                landmark = landmarks[idx]
                position = [landmark.x, landmark.y, landmark.z]

                # Calculate rotation relative to previous joint
                if idx > 0:
                    prev_landmark = landmarks[idx-1]
                    direction = np.array([landmark.x - prev_landmark.x,
                                          landmark.y - prev_landmark.y,
                                          landmark.z - prev_landmark.z])
                    rotation = self.calculate_rotation(direction)
                else:
                    rotation = [0, 0, 0, 1]

                finger_data[finger_name].append({
                    "position": position,
                    "rotation": rotation
                })

        return finger_data

    def calculate_rotation(self, direction):
        """Calculates a quaternion representing the rotation of a direction vector."""
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        #This is a simplified rotation calculation.  It assumes the "up"
        # direction is [0, 1, 0].  A more robust solution would use the hand's
        # orientation to determine the correct up vector.

        axis_x = np.array([1, 0, 0])
        axis_y = np.array([0, 1, 0])


        #Calculate rotation quaternion using cross product
        rotation_axis = np.cross(axis_x, direction)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-6)
        rotation_angle = np.arccos(np.dot(axis_x, direction))

        #Convert axis-angle to quaternion
        sin_half_angle = np.sin(rotation_angle / 2)
        cos_half_angle = np.cos(rotation_angle/2)

        x = rotation_axis[0] * sin_half_angle
        y = rotation_axis[1] * sin_half_angle
        z = rotation_axis[2] * sin_half_angle
        w = cos_half_angle

        return [x,y,z,w]



    def process_face_landmarks(self, landmarks):
        """Process face landmarks for face tracking with improved accuracy"""
        # より多くの顔のランドマークを活用
        face_data = {
            "eyes": {
                "left": self.process_eye_landmarks(landmarks, self.LEFT_EYE_INDICES),
                "right": self.process_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
            },
            "head": {
                "position": self.calculate_head_position(landmarks),
                "rotation": self.calculate_head_rotation(landmarks)
            },
            "jawline": self.process_jawline(landmarks),  # 顎のトラッキングを追加
            "cheeks": self.process_cheeks(landmarks)     # 頬のトラッキングを追加
        }
        return face_data

    def calculate_head_position(self, landmarks):
        """Calculate more accurate head position using multiple facial landmarks"""
        # 複数の顔のランドマークの重み付き平均を使用
        key_points = [
            (landmarks[1], 1.0),    # 鼻
            (landmarks[151], 0.8),  # 額
            (landmarks[454], 0.6),  # 左耳
            (landmarks[234], 0.6)   # 右耳
        ]
        
        weighted_sum = np.zeros(3)
        total_weight = 0
        
        for point, weight in key_points:
            weighted_sum += np.array([point.x, point.y, point.z]) * weight
            total_weight += weight
        
        return (weighted_sum / total_weight).tolist()

    def process_jawline(self, landmarks):
        """Process jawline landmarks for better facial expression tracking"""
        jawline_indices = [0, 17, 37, 267, 291, 314]  # 顎のランドマークインデックス
        return [
            [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
            for idx in jawline_indices
        ]

    def process_cheeks(self, landmarks):
        """Process cheek landmarks for improved expression detection"""
        cheek_indices = [123, 352]  # 頬のランドマークインデックス
        return [
            [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
            for idx in cheek_indices
        ]

    def process_eye_landmarks(self, landmarks, indices):
        """Process eye landmarks for eye tracking"""
        eye_points = []
        for idx in indices:
            landmark = landmarks[idx]
            eye_points.append([landmark.x, landmark.y, landmark.z])

        # Calculate eye openness and gaze direction
        eye_height = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        eye_width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        return {
            "openness": eye_height / eye_width,
            "gaze_direction": self.calculate_gaze_direction(eye_points)
        }

    def process_expressions(self, landmarks):
        """Process facial expressions with improved accuracy"""
        expressions = {
            "mouth_open": self.calculate_mouth_openness(landmarks),
            "smile": self.calculate_smile_value(landmarks),
            "brow_raise": self.calculate_brow_raise(landmarks),
            "eye_blink": self.calculate_eye_blink(landmarks),    # 目のまばたきを追加
            "cheek_puff": self.calculate_cheek_puff(landmarks),  # 頬の膨らみを追加
            "jaw_clench": self.calculate_jaw_clench(landmarks)   # 顎の噛みしめを追加
        }
        
        # 表情の正規化と組み合わせ効果の計算
        self.normalize_expressions(expressions)
        return expressions

    def normalize_expressions(self, expressions):
        """Normalize expression values and handle combinations"""
        # 値の範囲を0-1に正規化
        for key in expressions:
            expressions[key] = max(0.0, min(1.0, expressions[key]))
        
        # 表情の組み合わせ効果を考慮
        if expressions["smile"] > 0.5 and expressions["mouth_open"] > 0.3:
            expressions["laugh"] = min(1.0, expressions["smile"] * expressions["mouth_open"])

    def calculate_mouth_openness(self, landmarks):
        """Calculate mouth openness value"""
        upper_lip = landmarks[self.MOUTH_INDICES[0]]
        lower_lip = landmarks[self.MOUTH_INDICES[1]]
        return np.linalg.norm([upper_lip.x - lower_lip.x,
                               upper_lip.y - lower_lip.y,
                               upper_lip.z - lower_lip.z])

    def calculate_smile_value(self, landmarks):
        """Calculate smile value based on mouth corner positions"""
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        neutral_height = landmarks[0].y

        smile_value = ((left_corner.y + right_corner.y) / 2 - neutral_height) * 10
        return max(0, min(1, smile_value))

    def calculate_brow_raise(self, landmarks):
        """Calculate eyebrow raise value"""
        left_brow = landmarks[self.BROW_INDICES[0]]
        right_brow = landmarks[self.BROW_INDICES[1]]
        neutral_height = landmarks[9].y

        brow_raise = ((left_brow.y + right_brow.y) / 2 - neutral_height) * 10
        return max(0, min(1, brow_raise))

    def calculate_gaze_direction(self, eye_points):
        """Calculate eye gaze direction"""
        # Create a coordinate system from eye landmarks
        eye_center = np.mean(eye_points, axis=0)
        forward = np.array(eye_points[2]) - np.array(eye_points[5])
        up = np.array(eye_points[1]) - np.array(eye_points[5])

        # Normalize vectors
        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)

        # Calculate right vector
        right = np.cross(forward, up)

        # Return normalized gaze direction
        return forward.tolist()

    def calculate_head_rotation(self, landmarks):
        """Calculate head rotation from face landmarks"""
        # Use nose, forehead, and ears to calculate rotation
        nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        forehead = np.array([landmarks[151].x, landmarks[151].y, landmarks[151].z])
        left_ear = np.array([landmarks[454].x, landmarks[454].y, landmarks[454].z])
        right_ear = np.array([landmarks[234].x, landmarks[234].y, landmarks[234].z])

        # Calculate forward and up vectors
        forward = nose - forehead
        right = right_ear - left_ear

        # Calculate rotation matrix
        forward = forward / np.linalg.norm(forward)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Convert to quaternion
        rotation_matrix = np.vstack((right, up, forward)).T
        return self.matrix_to_quaternion(rotation_matrix)

    def matrix_to_quaternion(self, matrix):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                w = (matrix[2, 1] - matrix[1, 2]) / S
                x = 0.25 * S
                y = (matrix[0, 1] + matrix[1, 0]) / S
                z = (matrix[0, 2] + matrix[2, 0]) / S
            elif matrix[1, 1] > matrix[2, 2]:
                S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                w = (matrix[0, 2] - matrix[2, 0]) / S
                x = (matrix[0, 1] + matrix[1, 0]) / S
                y = 0.25 * S
                z = (matrix[1, 2] + matrix[2, 1]) / S
            else:
                S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                w = (matrix[1, 0] - matrix[0, 1]) / S
                x = (matrix[0, 2] + matrix[2, 0]) / S
                y = (matrix[1, 2] + matrix[2, 1]) / S
                z = 0.25 * S
        return [x, y, z, w]

    def send_vmc_data(self, tracking_data):
        """Send all tracking data via VMC Protocol"""
        # Send pose data
        if tracking_data["pose"]:
            for bone_name, bone_data in tracking_data["pose"]["bones"].items():
                self.vmc_client.send_message(
                    f"/VMC/Ext/Bone/Pos/{bone_name}",
                    bone_data["position"]+ bone_data["rotation"]
                )

        # Send hand data
        for hand_idx, hand_data in enumerate(tracking_data["hands"]):
            hand_prefix = f"Hand_{hand_idx}"  # Consider renaming to LeftHand/RightHand
            for finger_name, finger_joints in hand_data.items():
                for joint_idx, joint_data in enumerate(finger_joints):
                    self.vmc_client.send_message(
                        f"/VMC/Ext/Bone/Pos/{hand_prefix}/{finger_name}_{joint_idx}",
                        joint_data["position"] + joint_data["rotation"]
                    )

        # Send face data
        if tracking_data["face"]:
            # Send head position and rotation
            head_data = tracking_data["face"]["head"]
            self.vmc_client.send_message(
                "/VMC/Ext/Bone/Pos/Head",
                head_data["position"] + head_data["rotation"]
            )

            # Send eye tracking data
            for eye_name, eye_data in tracking_data["face"]["eyes"].items():
                self.vmc_client.send_message(
                    f"/VMC/Ext/Eye/{eye_name}",
                    [eye_data["openness"]] + eye_data["gaze_direction"]
                )

            # Send expression data
            for expr_name, expr_value in tracking_data["expressions"].items():
                self.vmc_client.send_message(
                    f"/VMC/Ext/Blend/Val/{expr_name}",
                    [expr_value]
                )

    def update_openvr_trackers(self, tracking_data):
        """Updates OpenVR trackers with the latest tracking data."""

        # --- Head Tracker ---
        if tracking_data["face"]:
            head_data = tracking_data["face"]["head"]
            head_position = head_data["position"]
            head_rotation = head_data["rotation"]

            # Convert MediaPipe coordinates to OpenVR coordinates.  This is
            # crucial for correct tracking in VR space.  MediaPipe's coordinate
            # system is different from OpenVR's.
            # MediaPipe:  +X right, +Y up, +Z away from camera
            # OpenVR:     +X right, +Y up, +Z towards camera (into the screen)
            openvr_head_position = [
                head_position[0] - 0.5,  # Center X (0.5 is the center of the image in normalized coords)
                -(head_position[1] - 0.5), # Center Y and invert (flip vertically)
                -head_position[2]  # Invert Z for depth
            ]

            if "head" not in self.openvr_trackers:
                # Create a new tracker if it doesn't exist
                try:
                    self.openvr_trackers["head"] = openvr.TrackedDeviceClass_GenericTracker
                    self.vr_system.addTrackedDevice("VMC_Head_Tracker", self.openvr_trackers["head"])
                except openvr.error_code.TrackedProp_Success as e: #Use the specific exception
                    print(f"Error adding head tracker: {e}")
                    return
            #Set pose for head tracker
            pose = openvr.TrackedDevicePose_t()
            pose.vVelocity.v[0] = 0  # No velocity for now
            pose.vVelocity.v[1] = 0
            pose.vVelocity.v[2] = 0
            pose.vAngularVelocity.v[0] = 0
            pose.vAngularVelocity.v[1] = 0
            pose.vAngularVelocity.v[2] = 0
            pose.eTrackingResult = openvr.TrackingResult_Running_OK
            pose.bDeviceIsConnected = True
            pose.bPoseIsValid = True

            # Set position
            pose.mDeviceToAbsoluteTracking[0][3] = openvr_head_position[0]
            pose.mDeviceToAbsoluteTracking[1][3] = openvr_head_position[1]
            pose.mDeviceToAbsoluteTracking[2][3] = openvr_head_position[2]

            # Set rotation (from quaternion)
            qw, qx, qy, qz = head_rotation[3], head_rotation[0], head_rotation[1], head_rotation[2] #Correct quaternion order
            pose.mDeviceToAbsoluteTracking[0][0] = 1 - 2 * (qy * qy + qz * qz)
            pose.mDeviceToAbsoluteTracking[0][1] = 2 * (qx * qy - qz * qw)
            pose.mDeviceToAbsoluteTracking[0][2] = 2 * (qx * qz + qy * qw)
            pose.mDeviceToAbsoluteTracking[1][0] = 2 * (qx * qy + qz * qw)
            pose.mDeviceToAbsoluteTracking[1][1] = 1 - 2 * (qx * qx + qz * qz)
            pose.mDeviceToAbsoluteTracking[1][2] = 2 * (qy * qz - qx * qw)
            pose.mDeviceToAbsoluteTracking[2][0] = 2 * (qx * qz - qy * qw)
            pose.mDeviceToAbsoluteTracking[2][1] = 2 * (qy * qz + qx * qw)
            pose.mDeviceToAbsoluteTracking[2][2] = 1 - 2 * (qx * qx + qy * qy)

            try:
                self.vr_system.updateTrackedDevicePose(self.openvr_trackers["head"], pose)
            except Exception as e:
                print(f"Error updating head tracker pose: {e}")



    def run(self):
        """Main loop"""
        try:
            while True:
                frame = self.process_frame()
                if frame is None:
                    break

                # Display the frame (for debugging)
                cv2.imshow('VMC Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
        openvr.shutdown()

if __name__ == "__main__":
    tracking_system = VMCTrackingSystem()
    tracking_system.run()