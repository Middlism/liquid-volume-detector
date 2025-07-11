import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRef, useState } from 'react';
import { Alert, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function CameraScreen({ navigation }) {
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const cameraRef = useRef(null);

  if (!permission) {
    // Camera permissions are still loading
    return <View style={styles.container}><Text style={styles.message}>Loading...</Text></View>;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const startRecording = async () => {
    if (cameraRef.current && !isRecording) {
      try {
        setIsRecording(true);
        const video = await cameraRef.current.recordAsync({
          maxDuration: 60,
        });
        console.log('Video recorded:', video.uri);
        Alert.alert('Success', 'Video recorded successfully!', [
          { text: 'OK', onPress: () => console.log('OK Pressed') }
        ]);
        // Here you would typically process the video or send it to your ML model
      } catch (error) {
        console.error('Error recording video:', error);
        Alert.alert('Error', 'Failed to record video');
        setIsRecording(false);
      }
    }
  };

  const stopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
      setIsRecording(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <CameraView 
        style={styles.camera} 
        facing={facing}
        ref={cameraRef}
        mode="video"
      >
        <View style={styles.overlay}>
          <View style={styles.topBar}>
            <TouchableOpacity onPress={() => navigation.goBack()}>
              <Ionicons name="arrow-back" size={30} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.instructionText}>
              Point camera at glass cup
            </Text>
          </View>

          <View style={styles.bottomBar}>
            <TouchableOpacity
              style={[styles.recordButton, isRecording && styles.recordingButton]}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <View style={[styles.recordButtonInner, isRecording && styles.recordButtonInnerRecording]} />
            </TouchableOpacity>
            <Text style={styles.recordText}>
              {isRecording ? 'Recording...' : 'Tap to record'}
            </Text>
          </View>
        </View>
      </CameraView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    color: '#fff',
    fontSize: 18,
    marginBottom: 10,
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 10,
    alignSelf: 'center',
    marginTop: 20,
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  submessage: {
    textAlign: 'center',
    color: '#ccc',
    fontSize: 14,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    paddingTop: 40,
  },
  instructionText: {
    color: '#fff',
    fontSize: 18,
    marginLeft: 20,
    flex: 1,
  },
  bottomBar: {
    alignItems: 'center',
    paddingBottom: 40,
  },
  recordButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderWidth: 5,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordingButton: {
    backgroundColor: 'rgba(255, 0, 0, 0.3)',
    borderColor: '#ff0000',
  },
  recordButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#fff',
  },
  recordButtonInnerRecording: {
    width: 30,
    height: 30,
    borderRadius: 5,
    backgroundColor: '#ff0000',
  },
  recordText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 10,
  },
});